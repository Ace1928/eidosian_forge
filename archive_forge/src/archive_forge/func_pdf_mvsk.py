import numpy as np
from numpy import poly1d, sqrt, exp
import scipy
from scipy import stats, special
from scipy.stats import distributions
from statsmodels.stats.moment_helpers import mvsk2mc, mc2mvsk
def pdf_mvsk(mvsk):
    """Return the Gaussian expanded pdf function given the list of 1st, 2nd
    moment and skew and Fisher (excess) kurtosis.



    Parameters
    ----------
    mvsk : list of mu, mc2, skew, kurt
        distribution is matched to these four moments

    Returns
    -------
    pdffunc : function
        function that evaluates the pdf(x), where x is the non-standardized
        random variable.


    Notes
    -----

    Changed so it works only if four arguments are given. Uses explicit
    formula, not loop.

    This implements a Gram-Charlier expansion of the normal distribution
    where the first 2 moments coincide with those of the normal distribution
    but skew and kurtosis can deviate from it.

    In the Gram-Charlier distribution it is possible that the density
    becomes negative. This is the case when the deviation from the
    normal distribution is too large.



    References
    ----------
    https://en.wikipedia.org/wiki/Edgeworth_series
    Johnson N.L., S. Kotz, N. Balakrishnan: Continuous Univariate
    Distributions, Volume 1, 2nd ed., p.30
    """
    N = len(mvsk)
    if N < 4:
        raise ValueError('Four moments must be given to approximate the pdf.')
    mu, mc2, skew, kurt = mvsk
    totp = poly1d(1)
    sig = sqrt(mc2)
    if N > 2:
        Dvals = _hermnorm(N + 1)
        C3 = skew / 6.0
        C4 = kurt / 24.0
        totp = totp - C3 * Dvals[3] + C4 * Dvals[4]

    def pdffunc(x):
        xn = (x - mu) / sig
        return totp(xn) * np.exp(-xn * xn / 2.0) / np.sqrt(2 * np.pi) / sig
    return pdffunc