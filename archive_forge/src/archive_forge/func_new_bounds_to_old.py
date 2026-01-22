import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
def new_bounds_to_old(lb, ub, n):
    """Convert the new bounds representation to the old one.

    The new representation is a tuple (lb, ub) and the old one is a list
    containing n tuples, ith containing lower and upper bound on a ith
    variable.
    If any of the entries in lb/ub are -np.inf/np.inf they are replaced by
    None.
    """
    lb = np.broadcast_to(lb, n)
    ub = np.broadcast_to(ub, n)
    lb = [float(x) if x > -np.inf else None for x in lb]
    ub = [float(x) if x < np.inf else None for x in ub]
    return list(zip(lb, ub))