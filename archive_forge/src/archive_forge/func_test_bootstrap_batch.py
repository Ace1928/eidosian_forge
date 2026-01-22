import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
@pytest.mark.parametrize('method', ['basic', 'percentile', 'BCa'])
@pytest.mark.parametrize('axis', [0, 1, 2])
def test_bootstrap_batch(method, axis):
    np.random.seed(0)
    x = np.random.rand(10, 11, 12)
    res1 = bootstrap((x,), np.mean, batch=None, method=method, random_state=0, axis=axis, n_resamples=100)
    res2 = bootstrap((x,), np.mean, batch=10, method=method, random_state=0, axis=axis, n_resamples=100)
    assert_equal(res2.confidence_interval.low, res1.confidence_interval.low)
    assert_equal(res2.confidence_interval.high, res1.confidence_interval.high)
    assert_equal(res2.standard_error, res1.standard_error)