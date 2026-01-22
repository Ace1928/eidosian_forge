import collections
import numpy as np
from scipy.optimize import OptimizeResult
from scipy.optimize._optimize import _check_unknown_options
from ._linesearch import _nonmonotone_line_search_cruz, _nonmonotone_line_search_cheng
def wrap_func(x):
    if nfev_list[0] >= maxfev:
        raise _NoConvergence()
    nfev_list[0] += 1
    x = x.reshape(x0_shape)
    F = np.asarray(func(x, *args)).ravel()
    f = fmerit(F)
    return (f, F)