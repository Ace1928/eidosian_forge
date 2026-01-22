import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_finite_precision_statistic(self):
    x = [1, 2, 4, 3]
    y = [2, 4, 6, 8]

    def statistic(x, y):
        return stats.pearsonr(x, y)[0]
    res = stats.permutation_test((x, y), statistic, vectorized=False, permutation_type='pairings')
    r, pvalue, null = (res.statistic, res.pvalue, res.null_distribution)
    correct_p = 2 * np.sum(null >= r - 1e-14) / len(null)
    assert pvalue == correct_p == 1 / 3