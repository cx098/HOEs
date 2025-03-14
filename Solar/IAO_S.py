import pandas as pd
import numpy as np
import warnings
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)
# 全局计数器
function_call_counter = 0

# 全局最优MSE
global_best_mse = np.inf

# 全局最优参数
global_best_params = None
# 数据加载
file_path = r"D:\Code\cx\data\Solar station site 1 (Nominal capacity-50MW).xlsx"
data = pd.read_excel(file_path)

# 日期格式转换
data['date'] = pd.to_datetime(data['Time'], format='%Y-%m-%d %H:%M:%S')

# 添加时间特征
data['month'] = data['date'].dt.month
data['day'] = data['date'].dt.day
data['hour'] = data['date'].dt.hour
data['minute'] = data['date'].dt.minute

# 特征和目标列
features = ['Total solar irradiance', 'Direct normal irradiance', 'Global horizontal irradiance',
            'Temperature', 'Atmosphere', 'month', 'day', 'hour', 'minute']
target = 'Power'

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
    """
    创建时间序列数据。
    - 主输入：目标变量的历史序列
    - 辅助输入：时间特征和其他数值型特征
    - 输出：目标变量的下一步预测值
    """
    X_main, X_aux, y = [], [], []

    for i in range(len(data) - time_steps):
        # 主输入是目标变量 Household 的历史时间步
        X_main.append(data['Power'].iloc[i:i + time_steps].values)

        # 辅助输入是其他特征的历史时间步
        X_aux.append(data[features].iloc[i:i + time_steps].values)

        # 输出是目标变量 Household 的下一步
        y.append(data['Power'].iloc[i + time_steps])

    return np.array(X_main), np.array(X_aux), np.array(y)


# 生成时间序列数据
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

def evaluate_model(params):
    global function_call_counter
    global global_best_mse
    global global_best_params

    function_call_counter += 1
    print(f"第 {function_call_counter} 次调用目标函数")

    params = params.flatten()  # 展平二维数组为一维
    learning_rate, dropout_rate = params[0], params[1]

    model = build_ts_lstme_model(time_steps, aux_feature_dim, r, dropout_rate, learning_rate)

    # 使用早停防止过拟合
    early_stopping = EarlyStopping(monitor='val_loss', patience=3)

    history = model.fit([X_main_train, X_aux_train], y_train, epochs=50, batch_size=64,
                        validation_data=([X_main_val, X_aux_val], y_val), callbacks=[early_stopping])

    # 预测并反标准化
    y_pred = model.predict([X_main_val, X_aux_val])
    y_pred_inv = target_scaler.inverse_transform(y_pred.reshape(-1, 1))
    y_val_inv = target_scaler.inverse_transform(y_val.reshape(-1, 1))

    # 计算评价指标（使用均方误差）
    mse_test = mean_squared_error(y_val_inv, y_pred_inv)
    r2_test = r2_score(y_val_inv, y_pred_inv)

    print(f"learning_rate:{learning_rate}, dropout_rate:{dropout_rate}")
    print(f"R2: {r2_test}, MSE: {mse_test}")

    # 更新并输出最优MSE和对应的参数
    if mse_test < global_best_mse:
        global_best_mse = mse_test
        global_best_params = (learning_rate, dropout_rate)

    print(f"当前最优解 - learning_rate: {global_best_params[0]}, Dropout: {global_best_params[1]}, MSE: {global_best_mse}")

    return mse_test


def IAO(N, Max_iter, LB, UB, Dim, fobj):
    # 初始化
    Info = initialization(N, Dim, UB, LB)
    New_Info = np.copy(Info)
    Ffun = np.zeros(Info.shape[0])
    Ffun_new = np.zeros(New_Info.shape[0])

    Best_Pinfo = np.zeros(Dim)
    Best_Finfo = float('inf')
    Historical_Info = np.zeros(Max_iter)

    # 主循环
    iter = 0
    while iter < Max_iter:
        # 检查边界并寻找最佳解
        for i in range(Info.shape[0]):
            Info[i, :] = np.clip(Info[i, :], LB, UB)
            Ffun[i] = fobj(Info[i, :])
            if Ffun[i] < Best_Finfo:
                Best_Finfo = Ffun[i]
                Best_Pinfo = Info[i, :]

        # 生成候选解
        for i in range(Info.shape[0]):
            r1, r2 = np.random.choice(range(Info.shape[0]), 2, replace=False)
            theta = np.random.uniform(-1, 1)
            New_Info[i, :] = Info[i, :] + (Info[r1, :] - Info[r2, :]) * theta
            New_Info[i, :] = np.clip(New_Info[i, :], LB, UB)

            Ffun_new[i] = fobj(New_Info[i, :])
            if Ffun_new[i] < Ffun[i]:
                Info[i, :] = New_Info[i, :]
                Ffun[i] = Ffun_new[i]
            if Ffun[i] < Best_Finfo:
                Best_Finfo = Ffun[i]
                Best_Pinfo = Info[i, :]

        # 过滤和评估信息
        for i in range(Info.shape[0]):
            r3, r4 = np.random.choice(range(Info.shape[0]), 2, replace=False)
            xi = 2 * (3.468 * np.random.rand() * (1 - np.random.rand() * np.cos(np.arccos(np.random.rand() * 1e4))) % 1)
            phi = (np.cos(2 * np.random.rand()) + 1) * (1 - (iter / Max_iter))
            gamma = np.sin((np.pi / 4) ** (iter / Max_iter)) + phi + (np.log(iter / Max_iter)) / 8
            delta = np.cos(np.pi / 2 * np.sqrt(abs(gamma))) / xi
            lam = 2 ** (np.sqrt(abs(gamma)) - 2)

            if np.random.rand() < 0.5:
                New_Info[i, :] = Info[i, :] - delta * np.random.rand() * (Info[r3, :] - Info[i, :])
            else:
                New_Info[i, :] = Info[i, :] + delta * np.random.rand() * (Info[r4, :] - Info[i, :])

            New_Info[i, :] = np.clip(New_Info[i, :], LB, UB)
            Ffun_new[i] = fobj(New_Info[i, :])
            if Ffun_new[i] < Ffun[i]:
                Info[i, :] = New_Info[i, :]
                Ffun[i] = Ffun_new[i]
            if Ffun[i] < Best_Finfo:
                Best_Finfo = Ffun[i]
                Best_Pinfo = Info[i, :]

            # 通过 Eq.(7) 更新位置
            if phi >= 0.5:
                New_Info[i, :] = Best_Pinfo * np.cos((np.pi / 2) * (np.sqrt(lam ** (1 / 3)))) - np.random.rand() * (
                            np.mean(Best_Pinfo) - Info[i, :])
            else:
                New_Info[i, :] = Best_Pinfo * np.cos((np.pi / 2) * (np.sqrt(lam ** (1 / 3)))) - (
                            np.random.rand() ** 2 * Best_Pinfo - (2 * np.random.rand() - 1) * Info[i, :]) * 0.8

            New_Info[i, :] = np.clip(New_Info[i, :], LB, UB)
            Ffun_new[i] = fobj(New_Info[i, :])
            if Ffun_new[i] < Ffun[i]:
                Info[i, :] = New_Info[i, :]
                Ffun[i] = Ffun_new[i]
            if Ffun[i] < Best_Finfo:
                Best_Finfo = Ffun[i]
                Best_Pinfo = Info[i, :]

        Historical_Info[iter] = Best_Finfo
        iter += 1

    return Best_Finfo, Best_Pinfo, Historical_Info


def initialization(N, Dim, UB, LB):
    UB = np.array(UB)
    LB = np.array(LB)
    B_no = UB.size  # number of boundaries

    if B_no == 1:
        Info = np.random.rand(N, Dim) * (UB - LB) + LB
    else:
        Info = np.zeros((N, Dim))
        for i in range(Dim):
            Ub_i = UB[i]
            Lb_i = LB[i]
            Info[:, i] = np.random.rand(N) * (Ub_i - Lb_i) + Lb_i

    return Info


def sumsqu(xx):
    return sum(i * x ** 2 for i, x in enumerate(xx, start=1))


def objective_function(params):
    # 参数范围
    lb = np.array([1e-6, 0.1])  # 变量下界
    ub = np.array([1e-4, 0.4])  # 变量上界

    if np.any(params < lb) or np.any(params > ub):
        print("参数越界，跳过这次训练")
        return np.inf  # 返回一个很大的值作为惩罚

    return evaluate_model(params)


main_feature_dim = 1  # 主要输入的特征维度，即AQI
aux_feature_dim = X_aux_train.shape[2]  # 辅助输入的特征维度
r = 24
N = 80
Max_iteration = 500
lb = np.array([1e-5, 0.1])  # 变量下界
ub = np.array([1e-2, 0.4])  # 变量上界
dim = 2

# Call the IVY function
Destination_fitness, Destination_position, Convergence_curve = IAO(N, Max_iteration, lb, ub, dim, objective_function)

# 打印结果
print("最优解 (学习率和Dropout率):", Destination_position)
print("最优适应度 (MSE):", Destination_fitness)
