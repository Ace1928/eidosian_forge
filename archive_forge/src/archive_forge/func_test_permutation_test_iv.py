import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_permutation_test_iv(self):

    def stat(x, y, axis):
        return stats.ttest_ind((x, y), axis).statistic
    message = 'each sample in `data` must contain two or more ...'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1]), stat)
    message = '`data` must be a tuple containing at least two samples'
    with pytest.raises(ValueError, match=message):
        permutation_test((1,), stat)
    with pytest.raises(TypeError, match=message):
        permutation_test(1, stat)
    message = '`axis` must be an integer.'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, axis=1.5)
    message = '`permutation_type` must be in...'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, permutation_type='ekki')
    message = '`vectorized` must be `True`, `False`, or `None`.'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, vectorized=1.5)
    message = '`n_resamples` must be a positive integer.'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, n_resamples=-1000)
    message = '`n_resamples` must be a positive integer.'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, n_resamples=1000.5)
    message = '`batch` must be a positive integer or None.'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, batch=-1000)
    message = '`batch` must be a positive integer or None.'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, batch=1000.5)
    message = '`alternative` must be in...'
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, alternative='ekki')
    message = "'herring' cannot be used to seed a"
    with pytest.raises(ValueError, match=message):
        permutation_test(([1, 2, 3], [1, 2, 3]), stat, random_state='herring')