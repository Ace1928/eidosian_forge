import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('dist_name', ('norm', 'logistic'))
@pytest.mark.parametrize('i', range(5))
def test_against_anderson(self, dist_name, i):

    def fun(a):
        rng = np.random.default_rng(394295467)
        x = stats.tukeylambda.rvs(a, size=100, random_state=rng)
        expected = stats.anderson(x, dist_name)
        return expected.statistic - expected.critical_values[i]
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        sol = root(fun, x0=0)
    assert sol.success
    a = sol.x[0]
    rng = np.random.default_rng(394295467)
    x = stats.tukeylambda.rvs(a, size=100, random_state=rng)
    expected = stats.anderson(x, dist_name)
    expected_stat = expected.statistic
    expected_p = expected.significance_level[i] / 100

    def statistic1d(x):
        return stats.anderson(x, dist_name).statistic
    dist_rvs = self.rvs(getattr(stats, dist_name).rvs, rng)
    with suppress_warnings() as sup:
        sup.filter(RuntimeWarning)
        res = monte_carlo_test(x, dist_rvs, statistic1d, n_resamples=1000, vectorized=False, alternative='greater')
    assert_allclose(res.statistic, expected_stat)
    assert_allclose(res.pvalue, expected_p, atol=2 * self.atol)