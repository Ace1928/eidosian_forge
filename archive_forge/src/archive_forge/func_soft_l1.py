from warnings import warn
import numpy as np
from numpy.linalg import norm
from scipy.sparse import issparse
from scipy.sparse.linalg import LinearOperator
from scipy.optimize import _minpack, OptimizeResult
from scipy.optimize._numdiff import approx_derivative, group_columns
from scipy.optimize._minimize import Bounds
from .trf import trf
from .dogbox import dogbox
from .common import EPS, in_bounds, make_strictly_feasible
def soft_l1(z, rho, cost_only):
    t = 1 + z
    rho[0] = 2 * (t ** 0.5 - 1)
    if cost_only:
        return
    rho[1] = t ** (-0.5)
    rho[2] = -0.5 * t ** (-1.5)