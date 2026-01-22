import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('alternative, expected_pvalue', (('less', 0.9708333333333), ('greater', 0.05138888888889), ('two-sided', 0.1027777777778)))
def test_against_spearmanr_in_R(self, alternative, expected_pvalue):
    """
        Results above from R cor.test, e.g.

        options(digits=16)
        x <- c(1.76405235, 0.40015721, 0.97873798,
               2.2408932, 1.86755799, -0.97727788)
        y <- c(2.71414076, 0.2488, 0.87551913,
               2.6514917, 2.01160156, 0.47699563)
        cor.test(x, y, method = "spearm", alternative = "t")
        """
    x = [1.76405235, 0.40015721, 0.97873798, 2.2408932, 1.86755799, -0.97727788]
    y = [2.71414076, 0.2488, 0.87551913, 2.6514917, 2.01160156, 0.47699563]
    expected_statistic = 0.7714285714285715

    def statistic1d(x):
        return stats.spearmanr(x, y).statistic
    res = permutation_test((x,), statistic1d, permutation_type='pairings', n_resamples=np.inf, alternative=alternative)
    assert_allclose(res.statistic, expected_statistic, rtol=self.rtol)
    assert_allclose(res.pvalue, expected_pvalue, atol=1e-13)