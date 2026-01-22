import numpy as np
def holt__(x, hw_args: HoltWintersArgs):
    """
    Simple Exponential Smoothing
    Minimization Function
    (,)
    """
    _, _, _, alphac, _, y_alpha = holt_init(x, hw_args)
    n = hw_args.n
    lvl = hw_args.lvl
    for i in range(1, n):
        lvl[i] = y_alpha[i - 1] + alphac * lvl[i - 1]
    return hw_args.y - lvl