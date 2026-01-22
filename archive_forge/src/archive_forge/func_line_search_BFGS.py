from warnings import warn
from scipy.optimize import _minpack2 as minpack2    # noqa: F401
from ._dcsrch import DCSRCH
import numpy as np
def line_search_BFGS(f, xk, pk, gfk, old_fval, args=(), c1=0.0001, alpha0=1):
    """
    Compatibility wrapper for `line_search_armijo`
    """
    r = line_search_armijo(f, xk, pk, gfk, old_fval, args=args, c1=c1, alpha0=alpha0)
    return (r[0], r[1], 0, r[2])