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
def test_against_fisher_exact(self, alternative):

    def statistic(x):
        return np.sum((x == 1) & (y == 1))
    rng = np.random.default_rng(6235696159000529929)
    x = (rng.random(7) > 0.6).astype(float)
    y = (rng.random(7) + 0.25 * x > 0.6).astype(float)
    tab = stats.contingency.crosstab(x, y)[1]
    res = permutation_test((x,), statistic, permutation_type='pairings', n_resamples=np.inf, alternative=alternative, random_state=rng)
    res2 = stats.fisher_exact(tab, alternative=alternative)
    assert_allclose(res.pvalue, res2[1])