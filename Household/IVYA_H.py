import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, LSTM, Bidirectional, Dropout, concatenate, BatchNormalization, Flatten
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import MinMaxScaler
import joblib
from tensorflow.keras.callbacks import EarlyStopping
import warnings

# 全局计数器
function_call_counter = 0

# 全局最优MSE
global_best_mse = np.inf

# 全局最优参数
global_best_params = None

# 忽略 FutureWarning
warnings.filterwarnings("ignore", category=FutureWarning)

# 数据加载
file_path = r"D:\Code\cx\data\household_power_consumption.csv"
data = pd.read_csv(file_path)

# 日期格式转换
data['Datetime'] = pd.to_datetime(data['Datetime'], format='%Y/%m/%d %H:%M:%S')

# 添加时间特征
data['month'] = data['Datetime'].dt.month
data['day'] = data['Datetime'].dt.day
data['hour'] = data['Datetime'].dt.hour
data['minute'] = data['Datetime'].dt.minute

# 特征和目标列
features = ['Global_reactive_power', 'Voltage', 'Global_intensity',
            'Sub_metering_1', 'Sub_metering_2', 'Sub_metering_3', 'month', 'day', 'hour', 'minute']
target = 'Global_active_power'

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
        # 主输入是目标变量 Global_active_power 的历史时间步
        X_main.append(data['Global_active_power'].iloc[i:i + time_steps].values)

        # 辅助输入是其他特征的历史时间步
        X_aux.append(data[features].iloc[i:i + time_steps].values)

        # 输出是目标变量 Global_active_power 的下一步
        y.append(data['Global_active_power'].iloc[i + time_steps])

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


def IVY(N, Max_iteration, lb, ub, dim, fobj):
    def CostFunction(x):
        return fobj(x)

    VarMin = np.array(lb)       # Variables Lower Bound
    VarMax = np.array(ub)       # Variables Upper Bound

    # IVYA Parameters
    MaxIt = Max_iteration        # Maximum Number of Iterations
    nPop = N                     # Population Size
    VarSize = (1, dim)           # Decision Variables Matrix Size

    # Initialization

    # Empty Plant Structure
    class Plant:
        def __init__(self):
            self.Position = None
            self.Cost = None
            self.GV = None

    pop = [Plant() for _ in range(nPop)]    # Initial Population Array

    for plant in pop:
        # Initialize Position
        # Eq.(1)
        plant.Position = np.random.uniform(VarMin, VarMax, VarSize)
        # Eq.(6)-upper condition
        plant.GV = plant.Position / (VarMax - VarMin)

        # Evaluation
        plant.Cost = CostFunction(plant.Position)

    # Initialize Best Cost History
    BestCosts = np.zeros(MaxIt)
    Convergence_curve = np.zeros(MaxIt)

    # Ivy Main Loop

    for it in range(MaxIt):
        # Get Best and Worst Cost Values
        Costs = [plant.Cost for plant in pop]
        BestCost = np.min(Costs)
        WorstCost = np.max(Costs)

        # Initialize new Population
        newpop = []

        for i in range(len(pop)):
            ii = i + 1
            if i == len(pop) - 1:
                ii = 0

            newsol = Plant()

            beta_1 = 1 + (np.random.rand() / 2)  # beta value in line 8 of "Algorithm 1 (Fig.2 in paper)"

            if pop[i].Cost < beta_1 * pop[0].Cost:
                # Eq.(5)-(6)
                newsol.Position = pop[i].Position + np.abs(np.random.randn(*VarSize)) * (pop[ii].Position - pop[i].Position) + np.random.randn(*VarSize) * pop[i].GV
            else:
                # Eq.(7)
                newsol.Position = pop[0].Position * (np.random.rand() + np.random.randn(*VarSize) * pop[i].GV)

            # Eq.(3)
            pop[i].GV = pop[i].GV * ((np.random.rand() ** 2) * np.random.randn(1, dim))

            # Ensure newsol.Position is within bounds
            newsol.Position = np.maximum(newsol.Position, VarMin)
            newsol.Position = np.minimum(newsol.Position, VarMax)
            # Eq.(8)
            newsol.GV = newsol.Position / (VarMax - VarMin)

            # Evaluate new population
            newsol.Cost = CostFunction(newsol.Position)

            newpop.append(newsol)

        # Merge Populations
        pop.extend(newpop)

        # Sort Population
        pop.sort(key=lambda plant: plant.Cost)

        # Competitive Exclusion (Delete Extra Members)
        if len(pop) > nPop:
            pop = pop[:nPop]

        # Store Best Solution Ever Found
        BestSol = pop[0]
        BestCosts[it] = BestSol.Cost
        Convergence_curve[it] = BestSol.Cost

    # Results
    Destination_fitness = pop[0].Cost
    Destination_position = pop[0].Position

    return Destination_fitness, Destination_position, Convergence_curve


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
Destination_fitness, Destination_position, Convergence_curve = IVY(N, Max_iteration, lb, ub, dim, objective_function)

# 打印结果
print("最优解 (学习率和Dropout率):", Destination_position)
print("最优适应度 (MSE):", Destination_fitness)
