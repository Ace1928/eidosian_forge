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
@deprecate_kwarg('unbiased', 'adjusted')
def pacf_ols(x: ArrayLike1D, nlags: int | None=None, efficient: bool=True, adjusted: bool=False) -> np.ndarray:
    """
    Calculate partial autocorrelations via OLS.

    Parameters
    ----------
    x : array_like
        Observations of time series for which pacf is calculated.
    nlags : int, optional
        Number of lags to return autocorrelation for. If not provided,
        uses min(10 * np.log10(nobs), nobs - 1).
    efficient : bool, optional
        If true, uses the maximum number of available observations to compute
        each partial autocorrelation. If not, uses the same number of
        observations to compute all pacf values.
    adjusted : bool, optional
        Adjust each partial autocorrelation by n / (n - lag).

    Returns
    -------
    ndarray
        The partial autocorrelations, (maxlag,) array corresponding to lags
        0, 1, ..., maxlag.

    See Also
    --------
    statsmodels.tsa.stattools.pacf
        Partial autocorrelation estimation.
    statsmodels.tsa.stattools.pacf_yw
         Partial autocorrelation estimation using Yule-Walker.
    statsmodels.tsa.stattools.pacf_burg
        Partial autocorrelation estimation using Burg"s method.

    Notes
    -----
    This solves a separate OLS estimation for each desired lag using method in
    [1]_. Setting efficient to True has two effects. First, it uses
    `nobs - lag` observations of estimate each pacf.  Second, it re-estimates
    the mean in each regression. If efficient is False, then the data are first
    demeaned, and then `nobs - maxlag` observations are used to estimate each
    partial autocorrelation.

    The inefficient estimator appears to have better finite sample properties.
    This option should only be used in time series that are covariance
    stationary.

    OLS estimation of the pacf does not guarantee that all pacf values are
    between -1 and 1.

    References
    ----------
    .. [1] Box, G. E., Jenkins, G. M., Reinsel, G. C., & Ljung, G. M. (2015).
       Time series analysis: forecasting and control. John Wiley & Sons, p. 66
    """
    x = array_like(x, 'x')
    nlags = int_like(nlags, 'nlags', optional=True)
    efficient = bool_like(efficient, 'efficient')
    adjusted = bool_like(adjusted, 'adjusted')
    nobs = x.shape[0]
    if nlags is None:
        nlags = max(min(int(10 * np.log10(nobs)), nobs // 2), 1)
    if nlags > nobs // 2:
        raise ValueError(f'nlags must be smaller than nobs // 2 ({nobs // 2})')
    pacf = np.empty(nlags + 1)
    pacf[0] = 1.0
    if efficient:
        xlags, x0 = lagmat(x, nlags, original='sep')
        xlags = add_constant(xlags)
        for k in range(1, nlags + 1):
            params = lstsq(xlags[k:, :k + 1], x0[k:], rcond=None)[0]
            pacf[k] = np.squeeze(params[-1])
    else:
        x = x - np.mean(x)
        xlags, x0 = lagmat(x, nlags, original='sep', trim='both')
        for k in range(1, nlags + 1):
            params = lstsq(xlags[:, :k], x0, rcond=None)[0]
            pacf[k] = np.squeeze(params[-1])
    if adjusted:
        pacf *= nobs / (nobs - np.arange(nlags + 1))
    return pacf