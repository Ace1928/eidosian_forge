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
@pytest.mark.parametrize('n_output', (1, 2, 3))
def test_make_ufunc(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
    else:
        func = np.mean
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    res = ufunc(ary)
    if n_output > 1:
        assert all((len(res_i) == 4 for res_i in res))
        assert all(((res_i == 1).all() for res_i in res))
    else:
        assert len(res) == 4
        assert (res == 1).all()