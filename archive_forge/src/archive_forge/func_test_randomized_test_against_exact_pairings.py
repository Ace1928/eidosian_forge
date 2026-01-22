import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_randomized_test_against_exact_pairings(self):
    alternative, rng = ('two-sided', self.rng)
    nx, ny, permutations = (8, 8, 40000)
    assert special.factorial(nx) > permutations
    x = stats.norm.rvs(size=nx)
    y = stats.norm.rvs(size=ny)
    data = [x]

    def statistic1d(x):
        return stats.pearsonr(x, y)[0]
    statistic = _resampling._vectorize_statistic(statistic1d)
    kwds = {'vectorized': True, 'permutation_type': 'samples', 'batch': 100, 'alternative': alternative, 'random_state': rng}
    res = permutation_test(data, statistic, n_resamples=permutations, **kwds)
    res2 = permutation_test(data, statistic, n_resamples=np.inf, **kwds)
    assert res.statistic == res2.statistic
    assert_allclose(res.pvalue, res2.pvalue, atol=0.01)