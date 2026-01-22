import numpy as np
from scipy.optimize import _lbfgsb, minimize
def fun_single_precision(x):
    x = x.astype(np.float32)
    return (np.sum(x ** 2), 2 * x)