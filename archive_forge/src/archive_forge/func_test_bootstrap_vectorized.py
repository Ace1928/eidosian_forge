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
@pytest.mark.parametrize('paired', [True, False])
def test_bootstrap_vectorized(method, axis, paired):
    np.random.seed(0)

    def my_statistic(x, y, z, axis=-1):
        return x.mean(axis=axis) + y.mean(axis=axis) + z.mean(axis=axis)
    shape = (10, 11, 12)
    n_samples = shape[axis]
    x = np.random.rand(n_samples)
    y = np.random.rand(n_samples)
    z = np.random.rand(n_samples)
    res1 = bootstrap((x, y, z), my_statistic, paired=paired, method=method, random_state=0, axis=0, n_resamples=100)
    assert res1.bootstrap_distribution.shape == res1.standard_error.shape + (100,)
    reshape = [1, 1, 1]
    reshape[axis] = n_samples
    x = np.broadcast_to(x.reshape(reshape), shape)
    y = np.broadcast_to(y.reshape(reshape), shape)
    z = np.broadcast_to(z.reshape(reshape), shape)
    res2 = bootstrap((x, y, z), my_statistic, paired=paired, method=method, random_state=0, axis=axis, n_resamples=100)
    assert_allclose(res2.confidence_interval.low, res1.confidence_interval.low)
    assert_allclose(res2.confidence_interval.high, res1.confidence_interval.high)
    assert_allclose(res2.standard_error, res1.standard_error)
    result_shape = list(shape)
    result_shape.pop(axis)
    assert_equal(res2.confidence_interval.low.shape, result_shape)
    assert_equal(res2.confidence_interval.high.shape, result_shape)
    assert_equal(res2.standard_error.shape, result_shape)