from statsmodels.compat.pandas import deprecate_kwarg
from collections.abc import Iterable
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.regression.linear_model import OLS, RegressionResultsWrapper
from statsmodels.stats._adnorm import anderson_statistic, normal_ad
from statsmodels.stats._lilliefors import (
from statsmodels.tools.validation import (
from statsmodels.tsa.tsatools import lagmat
def spec_white(resid, exog):
    """
    White's Two-Moment Specification Test

    Parameters
    ----------
    resid : array_like
        OLS residuals.
    exog : array_like
        OLS design matrix.

    Returns
    -------
    stat : float
        The test statistic.
    pval : float
        A chi-square p-value for test statistic.
    dof : int
        The degrees of freedom.

    See Also
    --------
    het_white
        White's test for heteroskedasticity.

    Notes
    -----
    Implements the two-moment specification test described by White's
    Theorem 2 (1980, p. 823) which compares the standard OLS covariance
    estimator with White's heteroscedasticity-consistent estimator. The
    test statistic is shown to be chi-square distributed.

    Null hypothesis is homoscedastic and correctly specified.

    Assumes the OLS design matrix contains an intercept term and at least
    one variable. The intercept is removed to calculate the test statistic.

    Interaction terms (squares and crosses of OLS regressors) are added to
    the design matrix to calculate the test statistic.

    Degrees-of-freedom (full rank) = nvar + nvar * (nvar + 1) / 2

    Linearly dependent columns are removed to avoid singular matrix error.

    References
    ----------
    .. [*] White, H. (1980). A heteroskedasticity-consistent covariance matrix
       estimator and a direct test for heteroscedasticity. Econometrica, 48:
       817-838.
    """
    x = array_like(exog, 'exog', ndim=2)
    e = array_like(resid, 'resid', ndim=1)
    if x.shape[1] < 2 or not np.any(np.ptp(x, 0) == 0.0):
        raise ValueError("White's specification test requires at least twocolumns where one is a constant.")
    i0, i1 = np.triu_indices(x.shape[1])
    exog = np.delete(x[:, i0] * x[:, i1], 0, 1)
    atol = 1e-14
    rtol = 1e-13
    tol = atol + rtol * exog.var(0)
    r = np.linalg.qr(exog, mode='r')
    mask = np.abs(r.diagonal()) < np.sqrt(tol)
    exog = exog[:, np.where(~mask)[0]]
    sqe = e * e
    sqmndevs = sqe - np.mean(sqe)
    d = np.dot(exog.T, sqmndevs)
    devx = exog - np.mean(exog, axis=0)
    devx *= sqmndevs[:, None]
    b = devx.T.dot(devx)
    stat = d.dot(np.linalg.solve(b, d))
    dof = devx.shape[1]
    pval = stats.chi2.sf(stat, dof)
    return (stat, pval, dof)