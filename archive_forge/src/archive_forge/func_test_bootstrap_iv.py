import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_bootstrap_iv():
    message = '`data` must be a sequence of samples.'
    with pytest.raises(ValueError, match=message):
        bootstrap(1, np.mean)
    message = '`data` must contain at least one sample.'
    with pytest.raises(ValueError, match=message):
        bootstrap(tuple(), np.mean)
    message = 'each sample in `data` must contain two or more observations...'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3], [1]), np.mean)
    message = 'When `paired is True`, all samples must have the same length '
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3], [1, 2, 3, 4]), np.mean, paired=True)
    message = '`vectorized` must be `True`, `False`, or `None`.'
    with pytest.raises(ValueError, match=message):
        bootstrap(1, np.mean, vectorized='ekki')
    message = '`axis` must be an integer.'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, axis=1.5)
    message = 'could not convert string to float'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, confidence_level='ni')
    message = '`n_resamples` must be a non-negative integer.'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=-1000)
    message = '`n_resamples` must be a non-negative integer.'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=1000.5)
    message = '`batch` must be a positive integer or None.'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, batch=-1000)
    message = '`batch` must be a positive integer or None.'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, batch=1000.5)
    message = '`method` must be in'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, method='ekki')
    message = "`bootstrap_result` must have attribute `bootstrap_distribution'"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, bootstrap_result=10)
    message = 'Either `bootstrap_result.bootstrap_distribution.size`'
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, n_resamples=0)
    message = "'herring' cannot be used to seed a"
    with pytest.raises(ValueError, match=message):
        bootstrap(([1, 2, 3],), np.mean, random_state='herring')