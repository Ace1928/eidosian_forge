from functools import wraps
def chuang_f1(individual):
    """Binary deceptive function from : Multivariate Multi-Model Approach for
    Globally Multimodal Problems by Chung-Yao Chuang and Wen-Lian Hsu.

    The function takes individual of 40+1 dimensions and has two global optima
    in [1,1,...,1] and [0,0,...,0].
    """
    total = 0
    if individual[-1] == 0:
        for i in range(0, len(individual) - 1, 4):
            total += inv_trap(individual[i:i + 4])
    else:
        for i in range(0, len(individual) - 1, 4):
            total += trap(individual[i:i + 4])
    return (total,)