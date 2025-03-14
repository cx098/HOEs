import numpy as np


def ED(fnc, funnum, D, NP, GEN, LBv, UBv):
    # 初始化上下界
    if np.isscalar(LBv):
        LB = np.ones(D) * LBv
        UB = np.ones(D) * UBv
    else:
        LB = np.array(LBv)
        UB = np.array(UBv)

    MAXFEV = NP * GEN
    eval_count = 0
    Evali = []
    ishow = 250
    ObjVal = np.zeros(NP)

    # 初始化种群
    Pop = LB + np.random.rand(NP, D) * (UB - LB)
    for i in range(NP):
        ObjVal[i] = fnc(Pop[i, :], funnum)
        eval_count += 1
    Evali.append(eval_count)

    # 最优个体初始化
    iBest = np.argmin(ObjVal)
    GlobalMin = ObjVal[iBest]
    Xbest = Pop[iBest, :]
    g = 1
    Fit_store = np.zeros(GEN)
    Fit_store[g - 1] = GlobalMin
    ubest = Xbest

    while g < GEN:
        ct = 1 - np.random.rand() * g / GEN
        if np.random.rand() < 0.1:
            # 使用公式 (2) 进行任务
            kd = np.argmax(ObjVal)
            SolD = LB + np.random.rand(D) * (UB - LB)
            f_d = fnc(SolD, funnum)
            eval_count += 1
            Pop[kd, :] = SolD
            ObjVal[kd] = f_d
            GlobalMin = f_d
            Xbest = SolD
        else:
            # 计算 c(t) 使用公式 (9)
            a = int(np.ceil(3 * ct))
            if a == 1:
                # 使用公式 (3) 结构
                for i in range(NP):
                    P = np.random.choice(NP, 3, replace=False)
                    h, p, k = P
                    SolC = (Pop[h, :] + Pop[p, :] + Pop[k, :]) / 3
                    SolC += 2 * (np.random.rand(D) - 0.5) * (Xbest - SolC)
                    SolC = check_bound(SolC, UB, LB)
                    f_c = fnc(SolC, funnum)
                    eval_count += 1
                    if f_c <= GlobalMin:
                        Xbest = SolC
                        GlobalMin = f_c

            elif a == 2:
                # 使用公式 (5) 技术
                for i in range(NP):
                    h = np.random.choice(NP)
                    SolB = Pop[i, :] + np.random.rand(D) * (Xbest - Pop[i, :]) + np.random.rand(D) * (Xbest - Pop[h, :])
                    SolB = check_bound(SolB, UB, LB)
                    f_b = fnc(SolB, funnum)
                    eval_count += 1
                    if f_b <= ObjVal[i]:
                        Pop[i, :] = SolB
                        ObjVal[i] = f_b
                        if f_b <= GlobalMin:
                            Xbest = SolB
                            GlobalMin = f_b

            elif a == 3:
                # 使用公式 (6) 人群
                for i in range(NP):
                    Change = np.random.randint(D)
                    A = np.random.choice(NP, 3, replace=False)
                    nb1, nb2, nb3 = A
                    SolA = Pop[i, :].copy()
                    SolA[Change] = Pop[i, Change] + (
                                Pop[i, Change] - (Pop[nb1, Change] + Pop[nb2, Change] + Pop[nb3, Change]) / 3) * (
                                               np.random.rand() - 0.5) * 2
                    SolA = check_bound(SolA, UB, LB)
                    f_a = fnc(SolA, funnum)
                    eval_count += 1
                    if f_a <= ObjVal[i]:
                        Pop[i, :] = SolA
                        ObjVal[i] = f_a
                        if f_a <= GlobalMin:
                            Xbest = SolA
                            GlobalMin = f_a

        if eval_count > MAXFEV:
            break

        if g % ishow == 0:
            print(f"Generation: {g}. Best f: {GlobalMin:.6f}")

        g += 1
        if GlobalMin < Fit_store[g - 2]:
            Fit_store[g - 1] = GlobalMin
            ubest = Xbest
        else:
            Fit_store[g - 1] = Fit_store[g - 2]
        Evali.append(eval_count)

    f = Fit_store[-1]
    X = ubest
    print(f"The best result: {f}")
    return f, X, eval_count, Fit_store, Evali


def check_bound(Sol, UB, LB):
    # 保证每个解在上下界范围内
    Sol = np.where(Sol < LB, LB + np.random.rand() * (UB - LB), Sol)
    Sol = np.where(Sol > UB, LB + np.random.rand() * (UB - LB), Sol)
    return Sol


# 定义目标函数，假设是简单的平方和问题
def objective_function(x, funnum):
    return np.sum(x**2)


# 参数设置
funnum = 1               # 目标函数编号
D = 2                   # 维度
NP = 30                  # 种群大小
GEN = 100                # 最大迭代次数
lb = np.array([-5.12] * D)
ub = np.array([5.12] * D)

# 调用ED函数
best_fitness, best_solution, evaluations, fitness_history, evaluation_history = ED(objective_function, funnum, D, NP, GEN, lb, ub)

# 输出结果
print("最佳适应度值:", best_fitness)
print("最佳解:", best_solution)
print("总评估次数:", evaluations)
