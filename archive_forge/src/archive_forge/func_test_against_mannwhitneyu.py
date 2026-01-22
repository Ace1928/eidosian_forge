import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('alternative', ('less', 'greater', 'two-sided'))
def test_against_mannwhitneyu(self, alternative):
    x = stats.uniform.rvs(size=(3, 5, 2), loc=0, random_state=self.rng)
    y = stats.uniform.rvs(size=(3, 5, 2), loc=0.05, random_state=self.rng)
    expected = stats.mannwhitneyu(x, y, axis=1, alternative=alternative)

    def statistic(x, y, axis):
        return stats.mannwhitneyu(x, y, axis=axis).statistic
    res = permutation_test((x, y), statistic, vectorized=True, n_resamples=np.inf, alternative=alternative, axis=1, random_state=self.rng)
    assert_allclose(res.statistic, expected.statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected.pvalue, rtol=self.rtol)