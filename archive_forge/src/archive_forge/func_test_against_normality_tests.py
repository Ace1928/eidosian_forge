import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('hypotest', (stats.skewtest, stats.kurtosistest))
@pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
@pytest.mark.parametrize('a', np.linspace(-2, 2, 5))
def test_against_normality_tests(self, hypotest, alternative, a):
    rng = np.random.default_rng(85723405)
    x = stats.skewnorm.rvs(a=a, size=150, random_state=rng)
    expected = hypotest(x, alternative=alternative)

    def statistic(x, axis):
        return hypotest(x, axis=axis).statistic
    norm_rvs = self.rvs(stats.norm.rvs, rng)
    res = monte_carlo_test(x, norm_rvs, statistic, vectorized=True, alternative=alternative)
    assert_allclose(res.statistic, expected.statistic)
    assert_allclose(res.pvalue, expected.pvalue, atol=self.atol)