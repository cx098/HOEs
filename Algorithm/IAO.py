import numpy as np


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


# 全局计数器
function_call_counter = 0
def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次迭代")
    return np.sum(x ** 2)


# 设置参数
dim = 2  # 维度
N = 20  # 种群大小
Max_iteration = 100  # 最大迭代次数
lb = [-5.12] * dim  # 下界
ub = [5.12] * dim  # 上界

# 调用BEO函数
A_Score, A_Pos, WAR_curve = IAO(N, Max_iteration, lb, ub, dim, objective_function)

# 输出结果
print("最优值：", A_Score)
print("最优位置：", A_Pos)
