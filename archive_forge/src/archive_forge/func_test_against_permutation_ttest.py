import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('alternative', ('less', 'greater'))
@pytest.mark.parametrize('permutations', (30, 1000000000.0))
@pytest.mark.parametrize('axis', (0, 1, 2))
def test_against_permutation_ttest(self, alternative, permutations, axis):
    x = np.arange(3 * 4 * 5).reshape(3, 4, 5)
    y = np.moveaxis(np.arange(4)[:, None, None], 0, axis)
    rng1 = np.random.default_rng(4337234444626115331)
    res1 = stats.ttest_ind(x, y, permutations=permutations, axis=axis, random_state=rng1, alternative=alternative)

    def statistic(x, y, axis):
        return stats.ttest_ind(x, y, axis=axis).statistic
    rng2 = np.random.default_rng(4337234444626115331)
    res2 = permutation_test((x, y), statistic, vectorized=True, n_resamples=permutations, alternative=alternative, axis=axis, random_state=rng2)
    assert_allclose(res1.statistic, res2.statistic, rtol=self.rtol)
    assert_allclose(res1.pvalue, res2.pvalue, rtol=self.rtol)