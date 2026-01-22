import warnings
from collections import namedtuple
import numpy as np
from scipy import optimize, stats
from scipy._lib._util import check_random_state
def fit_fun(data):
    params = np.apply_along_axis(fit_fun_1d, axis=-1, arr=data)
    if params.ndim > 1:
        params = params.T[..., np.newaxis]
    return params