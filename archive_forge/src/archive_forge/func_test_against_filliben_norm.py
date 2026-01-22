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
def test_against_filliben_norm(self):
    rng = np.random.default_rng(8024266430745011915)
    y = [6, 1, -4, 8, -2, 5, 0]
    known_params = {'loc': 0, 'scale': 1}
    res = stats.goodness_of_fit(stats.norm, y, known_params=known_params, statistic='filliben', random_state=rng)
    assert_allclose(res.statistic, 0.98538, atol=0.0001)
    assert 0.75 < res.pvalue < 0.9
    assert_allclose(res.statistic, 0.98540957187084, rtol=2e-05)
    assert_allclose(res.pvalue, 0.8875, rtol=0.002)