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
def test_gof_iv(self):
    dist = stats.norm
    x = [1, 2, 3]
    message = '`dist` must be a \\(non-frozen\\) instance of...'
    with pytest.raises(TypeError, match=message):
        goodness_of_fit(stats.norm(), x)
    message = '`data` must be a one-dimensional array of numbers.'
    with pytest.raises(ValueError, match=message):
        goodness_of_fit(dist, [[1, 2, 3]])
    message = '`statistic` must be one of...'
    with pytest.raises(ValueError, match=message):
        goodness_of_fit(dist, x, statistic='mm')
    message = '`n_mc_samples` must be an integer.'
    with pytest.raises(TypeError, match=message):
        goodness_of_fit(dist, x, n_mc_samples=1000.5)
    message = "'herring' cannot be used to seed a"
    with pytest.raises(ValueError, match=message):
        goodness_of_fit(dist, x, random_state='herring')