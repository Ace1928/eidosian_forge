from __future__ import annotations
from statsmodels.compat.numpy import lstsq
from statsmodels.compat.pandas import deprecate_kwarg
from statsmodels.compat.python import lzip
from statsmodels.compat.scipy import _next_regular
from typing import Literal, Union
import warnings
import numpy as np
from numpy.linalg import LinAlgError
import pandas as pd
from scipy import stats
from scipy.interpolate import interp1d
from scipy.signal import correlate
from statsmodels.regression.linear_model import OLS, yule_walker
from statsmodels.tools.sm_exceptions import (
from statsmodels.tools.tools import Bunch, add_constant
from statsmodels.tools.validation import (
from statsmodels.tsa._bds import bds
from statsmodels.tsa._innovations import innovations_algo, innovations_filter
from statsmodels.tsa.adfvalues import mackinnoncrit, mackinnonp
from statsmodels.tsa.tsatools import add_trend, lagmat, lagmat2ds
def levinson_durbin_pacf(pacf, nlags=None):
    """
    Levinson-Durbin algorithm that returns the acf and ar coefficients.

    Parameters
    ----------
    pacf : array_like
        Partial autocorrelation array for lags 0, 1, ... p.
    nlags : int, optional
        Number of lags in the AR model.  If omitted, returns coefficients from
        an AR(p) and the first p autocorrelations.

    Returns
    -------
    arcoefs : ndarray
        AR coefficients computed from the partial autocorrelations.
    acf : ndarray
        The acf computed from the partial autocorrelations. Array returned
        contains the autocorrelations corresponding to lags 0, 1, ..., p.

    References
    ----------
    .. [1] Brockwell, P.J. and Davis, R.A., 2016. Introduction to time series
        and forecasting. Springer.
    """
    pacf = array_like(pacf, 'pacf')
    nlags = int_like(nlags, 'nlags', optional=True)
    pacf = np.squeeze(np.asarray(pacf))
    if pacf[0] != 1:
        raise ValueError('The first entry of the pacf corresponds to lags 0 and so must be 1.')
    pacf = pacf[1:]
    n = pacf.shape[0]
    if nlags is not None:
        if nlags > n:
            raise ValueError('Must provide at least as many values from the pacf as the number of lags.')
        pacf = pacf[:nlags]
        n = pacf.shape[0]
    acf = np.zeros(n + 1)
    acf[1] = pacf[0]
    nu = np.cumprod(1 - pacf ** 2)
    arcoefs = pacf.copy()
    for i in range(1, n):
        prev = arcoefs[:-(n - i)].copy()
        arcoefs[:-(n - i)] = prev - arcoefs[i] * prev[::-1]
        acf[i + 1] = arcoefs[i] * nu[i - 1] + prev.dot(acf[1:-(n - i)][::-1])
    acf[0] = 1
    return (arcoefs, acf)