import numpy as np


# 定义 FLO 优化算法函数
def FLO(SearchAgents, Max_iterations, lowerbound, upperbound, dimension, fitness):
    lowerbound = np.ones(dimension) * lowerbound  # 将下界扩展为维度大小
    upperbound = np.ones(dimension) * upperbound  # 将上界扩展为维度大小

    # 初始化种群
    X = np.array([lowerbound + np.random.rand(dimension) * (upperbound - lowerbound) for _ in range(SearchAgents)])
    fit = np.array([fitness(ind) for ind in X])

    best_so_far = []
    average = []

    # 迭代优化过程
    for t in range(Max_iterations):
        # 更新最佳解
        Fbest = np.min(fit)
        blocation = np.argmin(fit)

        if t == 0:
            xbest = X[blocation, :].copy()
            fbest = Fbest
        elif Fbest < fbest:
            fbest = Fbest
            xbest = X[blocation, :].copy()

        # 遍历每个个体，更新位置
        for i in range(SearchAgents):
            # Phase 1: Hunting strategy (exploration)
            prey_position = np.where(fit < fit[i])[0]
            if len(prey_position) == 0:
                selected_prey = xbest
            else:
                if np.random.rand() < 0.5:
                    selected_prey = xbest
                else:
                    k = np.random.choice(prey_position)
                    selected_prey = X[k]

            I = np.random.randint(1, 3)
            X_new_P1 = X[i, :] + np.random.rand() * (selected_prey - I * X[i, :])
            X_new_P1 = np.clip(X_new_P1, lowerbound, upperbound)

            # 更新位置并计算适应度
            fit_new_P1 = fitness(X_new_P1)
            if fit_new_P1 < fit[i]:
                X[i, :] = X_new_P1
                fit[i] = fit_new_P1

            # Phase 2: Moving up the tree (exploitation)
            X_new_P2 = X[i, :] + (1 - 2 * np.random.rand()) * ((upperbound - lowerbound) / (t + 1))
            X_new_P2 = np.clip(X_new_P2, lowerbound / (t + 1), upperbound / (t + 1))

            # 更新位置并计算适应度
            fit_new_P2 = fitness(X_new_P2)
            if fit_new_P2 < fit[i]:
                X[i, :] = X_new_P2
                fit[i] = fit_new_P2

        best_so_far.append(fbest)
        average.append(np.mean(fit))

    return fbest, xbest, best_so_far


# 定义目标函数
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

# 调用 FLO 函数
A_Score, A_Pos, WAR_curve = FLO(N, Max_iteration, lb, ub, dim, objective_function)

# 输出结果
print("最优值：", A_Score)
print("最优位置：", A_Pos)
