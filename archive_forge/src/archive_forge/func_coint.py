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
def coint(y0, y1, trend='c', method='aeg', maxlag=None, autolag: str | None='aic', return_results=None):
    """
    Test for no-cointegration of a univariate equation.

    The null hypothesis is no cointegration. Variables in y0 and y1 are
    assumed to be integrated of order 1, I(1).

    This uses the augmented Engle-Granger two-step cointegration test.
    Constant or trend is included in 1st stage regression, i.e. in
    cointegrating equation.

    **Warning:** The autolag default has changed compared to statsmodels 0.8.
    In 0.8 autolag was always None, no the keyword is used and defaults to
    "aic". Use `autolag=None` to avoid the lag search.

    Parameters
    ----------
    y0 : array_like
        The first element in cointegrated system. Must be 1-d.
    y1 : array_like
        The remaining elements in cointegrated system.
    trend : str {"c", "ct"}
        The trend term included in regression for cointegrating equation.

        * "c" : constant.
        * "ct" : constant and linear trend.
        * also available quadratic trend "ctt", and no constant "n".

    method : {"aeg"}
        Only "aeg" (augmented Engle-Granger) is available.
    maxlag : None or int
        Argument for `adfuller`, largest or given number of lags.
    autolag : str
        Argument for `adfuller`, lag selection criterion.

        * If None, then maxlag lags are used without lag search.
        * If "AIC" (default) or "BIC", then the number of lags is chosen
          to minimize the corresponding information criterion.
        * "t-stat" based choice of maxlag.  Starts with maxlag and drops a
          lag until the t-statistic on the last lag length is significant
          using a 5%-sized test.
    return_results : bool
        For future compatibility, currently only tuple available.
        If True, then a results instance is returned. Otherwise, a tuple
        with the test outcome is returned. Set `return_results=False` to
        avoid future changes in return.

    Returns
    -------
    coint_t : float
        The t-statistic of unit-root test on residuals.
    pvalue : float
        MacKinnon"s approximate, asymptotic p-value based on MacKinnon (1994).
    crit_value : dict
        Critical values for the test statistic at the 1 %, 5 %, and 10 %
        levels based on regression curve. This depends on the number of
        observations.

    Notes
    -----
    The Null hypothesis is that there is no cointegration, the alternative
    hypothesis is that there is cointegrating relationship. If the pvalue is
    small, below a critical size, then we can reject the hypothesis that there
    is no cointegrating relationship.

    P-values and critical values are obtained through regression surface
    approximation from MacKinnon 1994 and 2010.

    If the two series are almost perfectly collinear, then computing the
    test is numerically unstable. However, the two series will be cointegrated
    under the maintained assumption that they are integrated. In this case
    the t-statistic will be set to -inf and the pvalue to zero.

    TODO: We could handle gaps in data by dropping rows with nans in the
    Auxiliary regressions. Not implemented yet, currently assumes no nans
    and no gaps in time series.

    References
    ----------
    .. [1] MacKinnon, J.G. 1994  "Approximate Asymptotic Distribution Functions
       for Unit-Root and Cointegration Tests." Journal of Business & Economics
       Statistics, 12.2, 167-76.
    .. [2] MacKinnon, J.G. 2010.  "Critical Values for Cointegration Tests."
       Queen"s University, Dept of Economics Working Papers 1227.
       http://ideas.repec.org/p/qed/wpaper/1227.html
    """
    y0 = array_like(y0, 'y0')
    y1 = array_like(y1, 'y1', ndim=2)
    trend = string_like(trend, 'trend', options=('c', 'n', 'ct', 'ctt'))
    string_like(method, 'method', options=('aeg',))
    maxlag = int_like(maxlag, 'maxlag', optional=True)
    autolag = string_like(autolag, 'autolag', optional=True, options=('aic', 'bic', 't-stat'))
    return_results = bool_like(return_results, 'return_results', optional=True)
    nobs, k_vars = y1.shape
    k_vars += 1
    if trend == 'n':
        xx = y1
    else:
        xx = add_trend(y1, trend=trend, prepend=False)
    res_co = OLS(y0, xx).fit()
    if res_co.rsquared < 1 - 100 * SQRTEPS:
        res_adf = adfuller(res_co.resid, maxlag=maxlag, autolag=autolag, regression='n')
    else:
        warnings.warn('y0 and y1 are (almost) perfectly colinear.Cointegration test is not reliable in this case.', CollinearityWarning, stacklevel=2)
        res_adf = (-np.inf,)
    if trend == 'n':
        crit = [np.nan] * 3
    else:
        crit = mackinnoncrit(N=k_vars, regression=trend, nobs=nobs - 1)
    pval_asy = mackinnonp(res_adf[0], regression=trend, N=k_vars)
    return (res_adf[0], pval_asy, crit)