import numpy as np
def holt_win_init(x, hw_args: HoltWintersArgs):
    """Initialization for the Holt Winters Seasonal Models"""
    hw_args.p[hw_args.xi.astype(bool)] = x
    if hw_args.transform:
        alpha, beta, gamma = to_restricted(hw_args.p, hw_args.xi, hw_args.bounds)
    else:
        alpha, beta, gamma = hw_args.p[:3]
    l0, b0, phi = hw_args.p[3:6]
    s0 = hw_args.p[6:]
    alphac = 1 - alpha
    betac = 1 - beta
    gammac = 1 - gamma
    y_alpha = alpha * hw_args.y
    y_gamma = gamma * hw_args.y
    hw_args.lvl[:] = 0
    hw_args.b[:] = 0
    hw_args.s[:] = 0
    hw_args.lvl[0] = l0
    hw_args.b[0] = b0
    hw_args.s[:hw_args.m] = s0
    return (alpha, beta, gamma, phi, alphac, betac, gammac, y_alpha, y_gamma)