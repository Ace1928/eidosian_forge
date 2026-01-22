import numpy as np
from scipy import stats
from statsmodels.stats.moment_helpers import cov2corr
from statsmodels.stats.base import HolderTuple
from statsmodels.tools.validation import array_like
def test_cov_diagonal(cov, nobs):
    """One sample hypothesis test that covariance matrix is diagonal matrix.

    The Null and alternative hypotheses are

    .. math::

       H0 &: \\Sigma = diag(\\sigma_i) \\\\
       H1 &: \\Sigma \\neq diag(\\sigma_i)

    where :math:`\\sigma_i` are the variances with unspecified values.

    Parameters
    ----------
    cov : array_like
        Covariance matrix of the data, estimated with denominator ``(N - 1)``,
        i.e. `ddof=1`.
    nobs : int
        number of observations used in the estimation of the covariance

    Returns
    -------
    res : instance of HolderTuple
        results with ``statistic, pvalue`` and other attributes like ``df``

    References
    ----------
    Rencher, Alvin C., and William F. Christensen. 2012. Methods of
    Multivariate Analysis: Rencher/Methods. Wiley Series in Probability and
    Statistics. Hoboken, NJ, USA: John Wiley & Sons, Inc.
    https://doi.org/10.1002/9781118391686.

    StataCorp, L. P. Stata Multivariate Statistics: Reference Manual.
    Stata Press Publication.
    """
    cov = np.asarray(cov)
    k = cov.shape[0]
    R = cov2corr(cov)
    statistic = -(nobs - 1 - (2 * k + 5) / 6) * _logdet(R)
    df = k * (k - 1) / 2
    pvalue = stats.chi2.sf(statistic, df)
    return HolderTuple(statistic=statistic, pvalue=pvalue, df=df, distr='chi2', null='diagonal')