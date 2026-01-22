import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
@Appender(constrain_stationary_multivariate_python.__doc__)
def constrain_stationary_multivariate(unconstrained, variance, transform_variance=False, prefix=None):
    use_list = type(unconstrained) is list
    if use_list:
        unconstrained = np.concatenate(unconstrained, axis=1)
    k_endog, order = unconstrained.shape
    order //= k_endog
    if order < 1:
        raise ValueError('Must have order at least 1')
    if k_endog < 1:
        raise ValueError('Must have at least 1 endogenous variable')
    if prefix is None:
        prefix, dtype, _ = find_best_blas_type([unconstrained, variance])
    dtype = prefix_dtype_map[prefix]
    unconstrained = np.asfortranarray(unconstrained, dtype=dtype)
    variance = np.asfortranarray(variance, dtype=dtype)
    sv_constrained = prefix_sv_map[prefix](unconstrained, order, k_endog)
    constrained, variance = prefix_pacf_map[prefix](sv_constrained, variance, transform_variance, order, k_endog)
    constrained = np.array(constrained, dtype=dtype)
    variance = np.array(variance, dtype=dtype)
    if use_list:
        constrained = [constrained[:k_endog, i * k_endog:(i + 1) * k_endog] for i in range(order)]
    return (constrained, variance)