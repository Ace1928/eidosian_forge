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
def test_filliben_property(self):
    rng = np.random.default_rng(8535677809395478813)
    x = rng.normal(loc=10, scale=0.5, size=100)
    res = stats.goodness_of_fit(stats.norm, x, statistic='filliben', random_state=rng)
    known_params = {'loc': 0, 'scale': 1}
    ref = stats.goodness_of_fit(stats.norm, x, known_params=known_params, statistic='filliben', random_state=rng)
    assert_allclose(res.statistic, ref.statistic, rtol=1e-15)