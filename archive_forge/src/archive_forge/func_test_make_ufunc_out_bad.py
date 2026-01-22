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
def test_make_ufunc_out_bad(n_output):
    if n_output == 3:
        func = lambda x: (np.mean(x), np.mean(x), np.mean(x))
        res = (np.empty((100,)), np.empty((100,)))
    elif n_output == 2:
        func = lambda x: (np.mean(x), np.mean(x))
        res = np.empty((100,))
    else:
        func = np.mean
        res = np.empty((100,))
    ufunc = make_ufunc(func, n_dims=1, n_output=n_output)
    ary = np.ones((4, 100))
    with pytest.raises(TypeError):
        ufunc(ary, out=res)