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
def test_stats_variance_2d():
    """Test for stats_variance_2d."""
    data_1 = np.random.randn(1000, 1000)
    data_2 = np.random.randn(1000000)
    school = load_arviz_data('centered_eight').posterior['mu'].values
    n_school = load_arviz_data('non_centered_eight').posterior['mu'].values
    assert np.allclose(np.var(school, ddof=1, axis=1), stats_variance_2d(school, ddof=1, axis=1))
    assert np.allclose(np.var(school, ddof=1, axis=0), stats_variance_2d(school, ddof=1, axis=0))
    assert np.allclose(np.var(n_school, ddof=1, axis=1), stats_variance_2d(n_school, ddof=1, axis=1))
    assert np.allclose(np.var(n_school, ddof=1, axis=0), stats_variance_2d(n_school, ddof=1, axis=0))
    assert np.allclose(np.var(data_2), stats_variance_2d(data_2))
    assert np.allclose(np.var(data_2, ddof=1), stats_variance_2d(data_2, ddof=1))
    assert np.allclose(np.var(data_1, axis=0), stats_variance_2d(data_1, axis=0))
    assert np.allclose(np.var(data_1, axis=1), stats_variance_2d(data_1, axis=1))
    assert np.allclose(np.var(data_1, axis=0, ddof=1), stats_variance_2d(data_1, axis=0, ddof=1))
    assert np.allclose(np.var(data_1, axis=1, ddof=1), stats_variance_2d(data_1, axis=1, ddof=1))