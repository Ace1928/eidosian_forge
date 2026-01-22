import os
import numpy as np
import numpy.testing as npt
from numpy.testing import assert_allclose, assert_equal
import pytest
from scipy import stats
from scipy.optimize import differential_evolution
from .test_continuous_basic import distcont
from scipy.stats._distn_infrastructure import FitError
from scipy.stats._distr_params import distdiscrete
from scipy.stats import goodness_of_fit
def test_mse_accuracy_2(self):
    rng = np.random.default_rng(9843212616816518964)
    dist = stats.uniform
    n = 10
    data = dist(3, 6).rvs(size=n, random_state=rng)
    bounds = {'loc': (0, 10), 'scale': (1e-08, 10)}
    res = stats.fit(dist, data, bounds=bounds, method='mse')
    x = np.sort(data)
    a = (n * x[0] - x[-1]) / (n - 1)
    b = (n * x[-1] - x[0]) / (n - 1)
    ref = (a, b - a)
    assert_allclose(res.params, ref, rtol=0.0001)