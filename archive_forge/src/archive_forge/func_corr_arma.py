import numpy as np
from statsmodels.regression.linear_model import yule_walker
from statsmodels.stats.moment_helpers import cov2corr
def corr_arma(k_vars, ar, ma):
    """create arma correlation matrix

    converts arma to autoregressive lag-polynomial with k_var lags

    ar and arma might need to be switched for generating residual process

    Parameters
    ----------
    ar : array_like, 1d
        AR lag-polynomial including 1 for lag 0
    ma : array_like, 1d
        MA lag-polynomial

    """
    from scipy.linalg import toeplitz
    from statsmodels.tsa.arima_process import arma2ar
    ar = arma2ar(ar, ma, lags=k_vars)[:k_vars]
    return toeplitz(ar)