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
def test_mse_accuracy_1(self):
    data = [2, 4]
    dist = stats.expon
    bounds = {'loc': (0, 0), 'scale': (1e-08, 10)}
    res_mle = stats.fit(dist, data, bounds=bounds, method='mle')
    assert_allclose(res_mle.params.scale, 3, atol=0.001)
    res_mse = stats.fit(dist, data, bounds=bounds, method='mse')
    assert_allclose(res_mse.params.scale, 3.915, atol=0.001)