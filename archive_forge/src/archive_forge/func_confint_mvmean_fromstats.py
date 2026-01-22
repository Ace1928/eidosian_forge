import numpy as np
from scipy import stats
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.stats.base import HolderTuple
from statsmodels.tools.validation import array_like
def confint_mvmean_fromstats(mean, cov, nobs, lin_transf=None, alpha=0.05, simult=False):
    """Confidence interval for linear transformation of a multivariate mean

    Either pointwise or simultaneous confidence intervals are returned.
    Data is provided in the form of summary statistics, mean, cov, nobs.

    Parameters
    ----------
    mean : ndarray
    cov : ndarray
    nobs : int
    lin_transf : array_like or None
        The linear transformation or contrast matrix for transforming the
        vector of means. If this is None, then the identity matrix is used
        which specifies the means themselves.
    alpha : float in (0, 1)
        confidence level for the confidence interval, commonly used is
        alpha=0.05.
    simult : bool
        If simult is False (default), then pointwise confidence interval is
        returned.
        Otherwise, a simultaneous confidence interval is returned.
        Warning: additional simultaneous confidence intervals might be added
        and the default for those might change.

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
    mean = np.asarray(mean)
    cov = np.asarray(cov)
    c = np.atleast_2d(lin_transf)
    k_vars = len(mean)
    if simult is False:
        values = c.dot(mean)
        quad_form = (c * cov.dot(c.T).T).sum(1)
        df = nobs - 1
        t_critval = stats.t.isf(alpha / 2, df)
        ci_diff = np.sqrt(quad_form / df) * t_critval
        low = values - ci_diff
        upp = values + ci_diff
    else:
        values = c.dot(mean)
        quad_form = (c * cov.dot(c.T).T).sum(1)
        factor = (nobs - 1) * k_vars / (nobs - k_vars) / nobs
        df = (k_vars, nobs - k_vars)
        f_critval = stats.f.isf(alpha, df[0], df[1])
        ci_diff = np.sqrt(factor * quad_form * f_critval)
        low = values - ci_diff
        upp = values + ci_diff
    return (low, upp, values)