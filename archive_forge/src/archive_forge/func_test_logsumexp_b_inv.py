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
@pytest.mark.parametrize('ary_dtype', [np.float64, np.float32, np.int32, np.int64])
@pytest.mark.parametrize('axis', [None, 0, 1, (-2, -1)])
@pytest.mark.parametrize('b_inv', [None, 0, 100, 101])
@pytest.mark.parametrize('keepdims', [True, False])
def test_logsumexp_b_inv(ary_dtype, axis, b_inv, keepdims):
    """Test ArviZ implementation of logsumexp.

    Test also compares against Scipy implementation.
    Case where b=None, they are equal. (N=len(ary))
    Second case where b=x, and x is 1/(number of elements), they are almost equal.

    Test tests against b_inv parameter.
    """
    ary = np.random.randn(100, 101).astype(ary_dtype)
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=True) is not None
    ary = ary.copy()
    assert _logsumexp(ary=ary, axis=axis, b_inv=b_inv, keepdims=keepdims, copy=False) is not None
    out = np.empty(5)
    assert _logsumexp(ary=np.random.randn(10, 5), axis=0, out=out) is not None
    if b_inv != 0:
        b_scipy = 1 / b_inv if b_inv is not None else None
        scipy_results = logsumexp(ary, b=b_scipy, axis=axis, keepdims=keepdims)
        arviz_results = _logsumexp(ary, b_inv=b_inv, axis=axis, keepdims=keepdims)
        assert_array_almost_equal(scipy_results, arviz_results)