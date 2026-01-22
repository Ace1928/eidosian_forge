import numpy as np
def holt_win_add_mul_dam(x, hw_args: HoltWintersArgs):
    """
    Additive and Additive Damped with Multiplicative Seasonal
    Minimization Function
    (A,M) & (Ad,M)
    """
    _, beta, _, phi, alphac, betac, gammac, y_alpha, y_gamma = holt_win_init(x, hw_args)
    lvl = hw_args.lvl
    b = hw_args.b
    s = hw_args.s
    m = hw_args.m
    for i in range(1, hw_args.n):
        lvl[i] = y_alpha[i - 1] / s[i - 1] + alphac * (lvl[i - 1] + phi * b[i - 1])
        b[i] = beta * (lvl[i] - lvl[i - 1]) + betac * phi * b[i - 1]
        s[i + m - 1] = y_gamma[i - 1] / (lvl[i - 1] + phi * b[i - 1]) + gammac * s[i - 1]
    return hw_args.y - (lvl + phi * b) * s[:-(m - 1)]