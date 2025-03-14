import numpy as np
from scipy.special import gamma  # 导入 gamma 函数


# 定义 Levy 飞行的函数
def levy_flight(dim, beta=1.5):
    sigma = (gamma(1 + beta) * np.sin(np.pi * beta / 2) /
             (gamma((1 + beta) / 2) * beta * 2 ** ((beta - 1) / 2))) ** (1 / beta)
    u = np.random.normal(0, sigma, size=dim)
    v = np.random.normal(0, 1, size=dim)
    step = u / np.abs(v) ** (1 / beta)
    return step


# 定义边界检查函数
def space_bound(X, ub, lb):
    return np.maximum(np.minimum(X, ub), lb)


# 初始化种群
def initialize_population(N, dim, lb, ub):
    PopPos = np.random.rand(N, dim) * (ub - lb) + lb
    return PopPos


# APO 算法的主函数
def APO(N, T, lb, ub, dim, fobj):
    PopPos = initialize_population(N, dim, lb, ub)
    PopFit = np.array([fobj(PopPos[i, :]) for i in range(N)])

    # 初始化最佳位置和最佳适应度值
    BestF = np.inf
    BestX = None

    for i in range(N):
        if PopFit[i] <= BestF:
            BestF = PopFit[i]
            BestX = PopPos[i, :].copy()

    curve = np.zeros(T)

    for It in range(T):
        for i in range(N):
            theta1 = (1 - It / T)
            B = 2 * np.log(1 / np.random.rand()) * theta1

            if B > 0.5:
                # 1. 航空飞行阶段
                while True:
                    RandInd = np.random.choice([j for j in range(N) if j != i])
                    step1 = PopPos[i, :] - PopPos[RandInd, :]
                    if np.linalg.norm(step1) != 0:
                        break

                Y = PopPos[i, :] + levy_flight(dim) * step1 + np.round(
                    0.5 * (0.05 + np.random.rand())) * np.random.randn(dim)
                R = np.random.rand(dim)
                step2 = (R - 0.5) * np.pi
                S = np.tan(step2)
                Z = Y * S

                # 边界检查
                Y = space_bound(Y, ub, lb)
                Z = space_bound(Z, ub, lb)

                NewPop = np.array([Y, Z])
                NewPopfit = np.array([fobj(Y), fobj(Z)])
                newPopPos = NewPop[np.argmin(NewPopfit)]

            else:
                # 2. 水下觅食阶段
                F = 0.5
                K = [j for j in range(N) if j != i]
                RandInd = np.random.choice(K, size=3, replace=False)
                f = (0.1 * (np.random.rand() - 1) * (T - It)) / T

                while True:
                    step1 = PopPos[RandInd[1], :] - PopPos[RandInd[2], :]
                    if np.linalg.norm(step1) != 0:
                        break

                if np.random.rand() < 0.5:
                    W = PopPos[RandInd[0], :] + F * step1
                else:
                    W = PopPos[RandInd[0], :] + F * levy_flight(dim) * step1

                Y = (1 + f) * W

                while True:
                    X_rand1 = PopPos[np.random.randint(N), :]
                    X_rand2 = PopPos[np.random.randint(N), :]
                    step2 = X_rand1 - X_rand2
                    if np.linalg.norm(step2) != 0:
                        break

                Epsilon = np.random.rand()
                if np.random.rand() < 0.5:
                    Z = PopPos[i, :] + Epsilon * step2
                else:
                    Z = PopPos[i, :] + F * levy_flight(dim) * step2

                NewPop = np.array([W, Y, Z])
                NewPopfit = np.array([fobj(W), fobj(Y), fobj(Z)])
                newPopPos = NewPop[np.argmin(NewPopfit)]

            newPopPos = space_bound(newPopPos, ub, lb)
            newPopFit = fobj(newPopPos)

            if newPopFit < PopFit[i]:
                PopFit[i] = newPopFit
                PopPos[i, :] = newPopPos

        for i in range(N):
            if PopFit[i] < BestF:
                BestF = PopFit[i]
                BestX = PopPos[i, :].copy()

        curve[It] = BestF

    return BestF, BestX, curve


# 定义目标函数
function_call_counter = 0


def objective_function(x):
    global function_call_counter
    function_call_counter += 1
    print(f"第 {function_call_counter} 次迭代")
    return np.sum(x ** 2)


# 设置参数
dim = 2
N = 20
T = 100
lb = np.array([-5.12] * dim)
ub = np.array([5.12] * dim)

# 调用 APO 算法
best_f, best_x, curve = APO(N, T, lb, ub, dim, objective_function)

print("最优值：", best_f)
print("最优位置：", best_x)
