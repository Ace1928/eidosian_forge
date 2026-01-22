import numpy as np
from scipy import stats
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.stats.base import HolderTuple
from statsmodels.tools.validation import array_like
def confint_mvmean(data, lin_transf=None, alpha=0.5, simult=False):
    """Confidence interval for linear transformation of a multivariate mean

    Either pointwise or simultaneous confidence intervals are returned.

    Parameters
    ----------
    data : array_like
        data with observations in rows and variables in columns
    lin_transf : array_like or None
        The linear transformation or contrast matrix for transforming the
        vector of means. If this is None, then the identity matrix is used
        which specifies the means themselves.
    alpha : float in (0, 1)
        confidence level for the confidence interval, commonly used is
        alpha=0.05.
    simult : bool
        If ``simult`` is False (default), then the pointwise confidence
        interval is returned.
        Otherwise, a simultaneous confidence interval is returned.
        Warning: additional simultaneous confidence intervals might be added
        and the default for those might change.

    Returns
    -------
    low : ndarray
        lower confidence bound on the linear transformed
    upp : ndarray
        upper confidence bound on the linear transformed
    values : ndarray
        mean or their linear transformation, center of the confidence region

    Notes
    -----
    Pointwise confidence interval is based on Johnson and Wichern
    equation (5-21) page 224.

    Simultaneous confidence interval is based on Johnson and Wichern
    Result 5.3 page 225.
    This looks like Sheffe simultaneous confidence intervals.

    Bonferroni corrected simultaneous confidence interval might be added in
    future

    References
    ----------
    Johnson, Richard A., and Dean W. Wichern. 2007. Applied Multivariate
    Statistical Analysis. 6th ed. Upper Saddle River, N.J: Pearson Prentice
    Hall.
    """
    x = np.asarray(data)
    nobs, k_vars = x.shape
    if lin_transf is None:
        lin_transf = np.eye(k_vars)
    mean = x.mean(0)
    cov = np.cov(x, rowvar=False, ddof=0)
    ci = confint_mvmean_fromstats(mean, cov, nobs, lin_transf=lin_transf, alpha=alpha, simult=simult)
    return ci