import pandas as pd
import numpy as np
import warnings
import os
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 数据加载
file_path = r"D:\Code\cx\data\Processed_Metro_Traffic_Volume.csv"
data = pd.read_csv(file_path)

# 日期格式转换
data['date_time'] = pd.to_datetime(data['date_time'], format='%Y-%m-%d %H:%M:%S')

# 独热编码 weather_main
encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
weather_encoded = encoder.fit_transform(data[['weather_main']])
weather_df = pd.DataFrame(weather_encoded, columns=encoder.get_feature_names_out(['weather_main']))

# 整合数据并去掉 weather_main 和 weather_description 列
data = pd.concat([data, weather_df], axis=1).drop(columns=['weather_main', 'weather_description'])

# 转换 holiday 列
data['holiday'] = data['holiday'].notna().astype(int)

# 添加时间特征
data['hour'] = data['date_time'].dt.hour

# 特征和目标列
features = ['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour'] + list(weather_df.columns)
target = 'traffic_volume'

# 划分训练集和验证集
train_size = int(0.7 * len(data))
train_data = data[:train_size].copy()
test_data = data[train_size:].copy()

# 初始化归一化器
feature_scaler = MinMaxScaler()
target_scaler = MinMaxScaler()

# 对训练集特征和目标进行归一化
train_features_scaled = feature_scaler.fit_transform(train_data[features])
train_target_scaled = target_scaler.fit_transform(train_data[[target]])

# 对测试集特征和目标进行归一化
test_features_scaled = feature_scaler.transform(test_data[features])
test_target_scaled = target_scaler.transform(test_data[[target]])

# 将数据重新转换为DataFrame
train_data.loc[:, features] = train_features_scaled
train_data.loc[:, [target]] = train_target_scaled
test_data.loc[:, features] = test_features_scaled
test_data.loc[:, [target]] = test_target_scaled

# 保存归一化器
joblib.dump(feature_scaler, "feature_scaler.pkl")
joblib.dump(target_scaler, "target_scaler.pkl")

# 创建时间序列数据
time_steps = 24


def create_sequences(data, time_steps):
    X_main, X_aux, y = [], [], []

    for i in range(len(data) - time_steps):
        # 主要输入是 traffic_volume
        X_main.append(data['traffic_volume'].iloc[i:i + time_steps].values)

        # 辅助输入包含假日、温度、降雨、降雪、云量、天气编码和小时
        X_aux.append(data[['holiday', 'temp', 'rain_1h', 'snow_1h', 'clouds_all', 'hour'] + weather_columns].iloc[
                     i:i + time_steps].values)

        # 目标值是 traffic_volume 的下一步
        y.append(data['traffic_volume'].iloc[i + time_steps])

    return np.array(X_main), np.array(X_aux), np.array(y)


# weather_columns 应包含 weather_main 独热编码生成的列
weather_columns = [col for col in data.columns if 'weather_main_' in col]

X_main_train, X_aux_train, y_train = create_sequences(train_data, time_steps)
X_main_val, X_aux_val, y_val = create_sequences(test_data, time_steps)

# 打印数据集的信息
print(f"训练数据 X_main_train 的形状: {X_main_train.shape}")
print(f"训练数据 X_aux_train 的形状: {X_aux_train.shape}")
print(f"训练数据 y_train 的形状: {y_train.shape}")
print(f"验证数据 X_main_val 的形状: {X_main_val.shape}")
print(f"验证数据 X_aux_val 的形状: {X_aux_val.shape}")
print(f"验证数据 y_val 的形状: {y_val.shape}")

# 定义TS-BiLSTM模型
def build_ts_lstme_model(time_steps, aux_feature_dim, r, dropout_rate, learning_rate):
    # 输入层
    main_input = Input(shape=(time_steps, 1), name='main_input')
    aux_input = Input(shape=(time_steps, aux_feature_dim), name='aux_input')

    # LSTM 层
    x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(main_input)
    x = Dropout(dropout_rate)(x)
    x = BatchNormalization()(x)

    # 在每个时间步进行预测
    outputs = []
    for i in range(0, time_steps, r):
        if i + r < time_steps:
            temp_x = Bidirectional(LSTM(32, return_sequences=True, activation='relu'))(x)
        else:
            temp_x = Bidirectional(LSTM(32, return_sequences=False, activation='relu'))(x)
        temp_x = Dense(16, activation='relu')(temp_x)
        temp_x = Dropout(dropout_rate)(temp_x)
        temp_x = Dense(1, activation='linear')(temp_x)  # 预测单步
        temp_x = Flatten()(temp_x)
        outputs.append(temp_x)

    # 合并所有预测输出
    concatenated_outputs = concatenate(outputs, axis=-1)

    # 辅助输入的处理
    aux_dense = Dense(16, activation='relu')(aux_input)
    aux_dense = Dropout(dropout_rate)(aux_dense)
    aux_dense = Flatten()(aux_dense)

    # 最终拼接和输出
    final_concat = concatenate([concatenated_outputs, aux_dense], axis=-1)
    final_output = Dense(1, activation='linear', name='final_output')(final_concat)  # 预测单步的 AQI

    # 编译模型
    optimizer = Adam(learning_rate=learning_rate)
    model = Model(inputs=[main_input, aux_input], outputs=final_output)
    model.compile(optimizer=optimizer, loss='mean_squared_error', metrics=['mae'])

    return model


# 构建和训练模型
main_feature_dim = 1  # 主要输入的特征维度，即AQI
aux_feature_dim = X_aux_train.shape[2]  # 辅助输入的特征维度
r = 24
learning_rate = 0.000069104130784973
dropout_rate = 0.219833119504691
# 保存数据到CSV文件
name = "HOES"
save_path = r"D:\Code\cx\Traffic"
train_save_path = r"D:\Code\cx\Traffic\train"

model = build_ts_lstme_model(time_steps, aux_feature_dim, r, dropout_rate, learning_rate)

# 使用早停防止过拟合
early_stopping = EarlyStopping(monitor='val_loss', patience=3)

# history = model.fit([X_main_train, X_aux_train], y_train, epochs=50, batch_size=32, validation_data=([X_main_val, X_aux_val], y_val), callbacks=[early_stopping])
history = model.fit([X_main_train, X_aux_train], y_train, epochs=50, batch_size=64, validation_data=([X_main_val, X_aux_val], y_val))

# 计算IA (Index of Agreement)
def index_of_agreement(y_true, y_pred):
    numerator = np.sum((y_true - y_pred) ** 2)
    denominator = np.sum((np.abs(y_pred - np.mean(y_true)) + np.abs(y_true - np.mean(y_true))) ** 2)
    return 1 - (numerator / denominator)

# 计算训练集的预测值
y_train_pred = model.predict([X_main_train, X_aux_train])

# 计算验证集的预测值
y_pred = model.predict([X_main_val, X_aux_val])

# 修正预测数据的维度
y_train_pred = y_train_pred.flatten()  # 将二维数组转换为一维数组
y_pred = y_pred.flatten()  # 将二维数组转换为一维数组

# 打印维度以检查
print("y_val shape:", y_val.shape)
print(y_val)
print("y_pred shape:", y_pred.shape)
print(y_pred)

# 计算训练集的评价指标
mae_train = mean_absolute_error(y_train, y_train_pred)
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
r2_train = r2_score(y_train, y_train_pred)
ia_train = index_of_agreement(y_train, y_train_pred)

# 计算验证集的评价指标
mae_test = mean_absolute_error(y_val, y_pred)
mse_test = mean_squared_error(y_val, y_pred)
rmse_test = np.sqrt(mse_test)
r2_test = r2_score(y_val, y_pred)
ia_test = index_of_agreement(y_val, y_pred)

# 打印评价指标
print(f"训练集 MSE: {mse_train:.3f}")
print(f"训练集 RMSE: {rmse_train:.3f}")
print(f"训练集 MAE: {mae_train:.3f}")
print(f"训练集 R²: {r2_train:.3f}")
print(f"训练集 IA: {ia_train:.3f}")
print("***************************************************")
print(f"测试集 MSE: {mse_test:.3f}")
print(f"测试集 RMSE: {rmse_test:.3f}")
print(f"测试集 MAE: {mae_test:.3f}")
print(f"测试集 R²: {r2_test:.3f}")
print(f"测试集 IA: {ia_test:.3f}")

# 保存训练和验证损失到 DataFrame
loss_df = pd.DataFrame({
    'epoch': range(1, len(history.history['loss']) + 1),
    'train_loss': history.history['loss'],
    'val_loss': history.history['val_loss']
})

# 添加评价指标到 DataFrame
metrics_df = pd.DataFrame({
    'mae_train': [mae_train],
    'mse_train': [mse_train],
    'rmse_train': [rmse_train],
    'r2_train': [r2_train],
    'ia_train': [ia_train],
    'mae_test': [mae_test],
    'mse_test': [mse_test],
    'rmse_test': [rmse_test],
    'r2_test': [r2_test],
    'ia_test': [ia_test]
})

# 合并损失数据和评价指标数据
full_df = pd.concat([loss_df, metrics_df], axis=1)

os.makedirs(save_path, exist_ok=True)  # 确保路径存在
os.makedirs(train_save_path, exist_ok=True)  # 确保路径存在

csv_file_path = os.path.join(train_save_path, f"{name}.csv")  # 生成完整的CSV文件路径
full_df.to_csv(csv_file_path, index=False)  # 保存为CSV文件

# 保存模型
model_file_path = os.path.join(train_save_path, f"{name}.h5")  # 生成完整的模型文件路径
model.save(model_file_path)  # 保存模型

print("训练和验证损失及评价指标已保存到 CSV 文件。")

# 反归一化数据
y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))
y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))

# 保存真实值和预测值到 CSV 文件
output_df = pd.DataFrame({
    'Actual Values': y_val_inv.flatten(),  # 将二维数组转换为一维
    'Predicted Values': y_pred_inv.flatten()
})
output_file = os.path.join(save_path, f"{name}_Actual_vs_Predicted.csv")
output_df.to_csv(output_file, index=False)
print(f"真实值和预测值已保存到 CSV 文件: {output_file}")

# 解决中文显示问题
plt.rcParams['font.sans-serif'] = ['SimHei']  # 设置字体为黑体
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 绘制loss下降图并保存
plt.figure(figsize=(12, 6))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path, f"{name}_Training_and_Validation_Loss.png"))
plt.close()

# 绘制预测值与真实值对比图（前100个点）并保存
plt.figure(figsize=(12, 6))
plt.plot(y_val_inv[:100], label='Actual Values', color='blue')
plt.plot(y_pred_inv[1:101], label='Predicted Values', color='red')
plt.title(f'Comparison of Predicted and Actual Values (First 100 Points)')
plt.xlabel('Time Steps')
plt.ylabel('traffic_volume')
plt.legend()
plt.grid(True)
plt.savefig(os.path.join(save_path, f"{name}_Comparison_Predicted_vs_Actual_Values_First_100.png"))
plt.close()