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
@pytest.mark.parametrize('out_shape', ((1, 2), (1, 2, 3), (2, 3, 4, 5)))
@pytest.mark.parametrize('input_dim', ((4, 100), (4, 100, 3), (4, 100, 4, 5)))
def test_wrap_ufunc_out_shape(out_shape, input_dim):
    func = lambda x: np.random.rand(*out_shape)
    ary = np.ones(input_dim)
    res = wrap_xarray_ufunc(func, ary, func_kwargs={'out_shape': out_shape}, ufunc_kwargs={'n_dims': 1})
    assert res.shape == (*ary.shape[:-1], *out_shape)