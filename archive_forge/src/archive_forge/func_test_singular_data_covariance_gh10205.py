from scipy import stats, linalg, integrate
import numpy as np
from numpy.testing import (assert_almost_equal, assert_, assert_equal,
import pytest
from pytest import raises as assert_raises
def test_singular_data_covariance_gh10205():
    rng = np.random.default_rng(2321583144339784787)
    mu = np.array([1, 10, 20])
    sigma = np.array([[4, 10, 0], [10, 25, 0], [0, 0, 100]])
    data = rng.multivariate_normal(mu, sigma, 1000)
    try:
        stats.gaussian_kde(data.T)
    except linalg.LinAlgError:
        msg = 'The data appears to lie in a lower-dimensional subspace...'
        with assert_raises(linalg.LinAlgError, match=msg):
            stats.gaussian_kde(data.T)