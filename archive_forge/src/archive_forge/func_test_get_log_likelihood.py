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
def test_get_log_likelihood():
    idata = from_dict(log_likelihood={'y1': np.random.normal(size=(4, 100, 6)), 'y2': np.random.normal(size=(4, 100, 8))})
    lik1 = get_log_likelihood(idata, 'y1')
    lik2 = get_log_likelihood(idata, 'y2')
    assert lik1.shape == (4, 100, 6)
    assert lik2.shape == (4, 100, 8)