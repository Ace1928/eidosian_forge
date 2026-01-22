import numpy as np
from ._hessian_update_strategy import BFGS
from ._differentiable_functions import (
from ._optimize import OptimizeWarning
from warnings import warn, catch_warnings, simplefilter, filterwarnings
from scipy.sparse import issparse
def j_ineq(x):
    dy = np.zeros((n_bound_below + n_bound_above, len(x0)))
    dy_all = jac(x)
    if issparse(dy_all):
        dy_all = dy_all.toarray()
    dy_all = np.atleast_2d(dy_all)
    dy[:n_bound_below, :] = dy_all[i_bound_below]
    dy[n_bound_below:, :] = -dy_all[i_bound_above]
    return dy