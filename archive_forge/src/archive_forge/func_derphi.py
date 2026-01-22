from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def derphi(alpha):
    gc[0] += 1
    gval[0] = fprime(xk + alpha * pk, *args)
    gval_alpha[0] = alpha
    return np.dot(gval[0], pk)