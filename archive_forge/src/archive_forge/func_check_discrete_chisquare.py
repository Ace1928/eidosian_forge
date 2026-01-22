import numpy.testing as npt
from numpy.testing import assert_allclose
import numpy as np
import pytest
from scipy import stats
from .common_tests import (check_normalization, check_moment,
from scipy.stats._distr_params import distdiscrete, invdistdiscrete
from scipy.stats._distn_infrastructure import rv_discrete_frozen
def check_discrete_chisquare(distfn, arg, rvs, alpha, msg):
    """Perform chisquare test for random sample of a discrete distribution

    Parameters
    ----------
    distname : string
        name of distribution function
    arg : sequence
        parameters of distribution
    alpha : float
        significance level, threshold for p-value

    Returns
    -------
    result : bool
        0 if test passes, 1 if test fails

    """
    wsupp = 0.05
    _a, _b = distfn.support(*arg)
    lo = int(max(_a, -1000))
    high = int(min(_b, 1000)) + 1
    distsupport = range(lo, high)
    last = 0
    distsupp = [lo]
    distmass = []
    for ii in distsupport:
        current = distfn.cdf(ii, *arg)
        if current - last >= wsupp - 1e-14:
            distsupp.append(ii)
            distmass.append(current - last)
            last = current
            if current > 1 - wsupp:
                break
    if distsupp[-1] < _b:
        distsupp.append(_b)
        distmass.append(1 - last)
    distsupp = np.array(distsupp)
    distmass = np.array(distmass)
    histsupp = distsupp + 1e-08
    histsupp[0] = _a
    freq, hsupp = np.histogram(rvs, histsupp)
    chis, pval = stats.chisquare(np.array(freq), len(rvs) * distmass)
    npt.assert_(pval > alpha, f'chisquare - test for {msg} at arg = {str(arg)} with pval = {str(pval)}')