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
def test_data_iv(self):
    message = '`data` must be exactly one-dimensional.'
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, [[1, 2, 3]], self.shape_bounds_a)
    message = 'All elements of `data` must be finite numbers.'
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, [1, 2, 3, np.nan], self.shape_bounds_a)
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, [1, 2, 3, np.inf], self.shape_bounds_a)
    with pytest.raises(ValueError, match=message):
        stats.fit(self.dist, ['1', '2', '3'], self.shape_bounds_a)