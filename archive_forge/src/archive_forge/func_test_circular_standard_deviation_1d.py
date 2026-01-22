import numpy as np
import pytest
from numpy.testing import assert_array_almost_equal
from scipy.special import logsumexp
from scipy.stats import circstd
from ...data import from_dict, load_arviz_data
from ...stats.density_utils import histogram
from ...stats.stats_utils import (
from ...stats.stats_utils import logsumexp as _logsumexp
from ...stats.stats_utils import make_ufunc, not_valid, stats_variance_2d, wrap_xarray_ufunc
@pytest.mark.parametrize('data', (np.random.randn(100), np.random.randn(100, 100), np.random.randn(100, 100, 100)))
def test_circular_standard_deviation_1d(data):
    high = 8
    low = 4
    assert np.allclose(_circular_standard_deviation(data, high=high, low=low), circstd(data, high=high, low=low))