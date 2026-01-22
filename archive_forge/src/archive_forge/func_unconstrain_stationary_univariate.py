import numpy as np
from scipy.linalg import solve_sylvester
import pandas as pd
from statsmodels.compat.pandas import Appender
from statsmodels.tools.data import _is_using_pandas
from scipy.linalg.blas import find_best_blas_type
from . import (_initialization, _representation, _kalman_filter,
def unconstrain_stationary_univariate(constrained):
    """
    Transform constrained parameters used in likelihood evaluation
    to unconstrained parameters used by the optimizer

    Parameters
    ----------
    constrained : ndarray
        Constrained parameters of, e.g., an autoregressive or moving average
        component, to be transformed to arbitrary parameters used by the
        optimizer.

    Returns
    -------
    unconstrained : ndarray
        Unconstrained parameters used by the optimizer, to be transformed to
        stationary coefficients of, e.g., an autoregressive or moving average
        component.

    References
    ----------
    .. [*] Monahan, John F. 1984.
       "A Note on Enforcing Stationarity in
       Autoregressive-moving Average Models."
       Biometrika 71 (2) (August 1): 403-404.
    """
    n = constrained.shape[0]
    y = np.zeros((n, n), dtype=constrained.dtype)
    y[n - 1:] = -constrained
    for k in range(n - 1, 0, -1):
        for i in range(k):
            y[k - 1, i] = (y[k, i] - y[k, k] * y[k, k - i - 1]) / (1 - y[k, k] ** 2)
    r = y.diagonal()
    x = r / (1 - r ** 2) ** 0.5
    return x