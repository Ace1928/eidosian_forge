import numpy as np
import pytest
from scipy.stats import bootstrap, monte_carlo_test, permutation_test
from numpy.testing import assert_allclose, assert_equal, suppress_warnings
from scipy import stats
from scipy import special
from .. import _resampling as _resampling
from scipy._lib._util import rng_integers
from scipy.optimize import root
def test_BCa_acceleration_against_reference():
    y = np.array([10, 27, 31, 40, 46, 50, 52, 104, 146])
    z = np.array([16, 23, 38, 94, 99, 141, 197])

    def statistic(z, y, axis=0):
        return np.mean(z, axis=axis) - np.mean(y, axis=axis)
    data = [z, y]
    res = stats.bootstrap(data, statistic)
    axis = -1
    alpha = 0.95
    theta_hat_b = res.bootstrap_distribution
    batch = 100
    _, _, a_hat = _resampling._bca_interval(data, statistic, axis, alpha, theta_hat_b, batch)
    assert_allclose(a_hat, 0.011008228344026734)