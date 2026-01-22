import pytest
import numpy as np
from numpy.random import seed
from numpy.testing import assert_allclose
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
from scipy.linalg import cossin, get_lapack_funcs
@pytest.mark.parametrize('dtype_', DTYPES)
@pytest.mark.parametrize('m, p, q', [(2, 1, 1), (3, 2, 1), (3, 1, 2), (4, 2, 2), (4, 1, 2), (40, 12, 20), (40, 30, 1), (40, 1, 30), (100, 50, 1), (100, 50, 50)])
@pytest.mark.parametrize('swap_sign', [True, False])
def test_cossin(dtype_, m, p, q, swap_sign):
    seed(1234)
    if dtype_ in COMPLEX_DTYPES:
        x = np.array(unitary_group.rvs(m), dtype=dtype_)
    else:
        x = np.array(ortho_group.rvs(m), dtype=dtype_)
    u, cs, vh = cossin(x, p, q, swap_sign=swap_sign)
    assert_allclose(x, u @ cs @ vh, rtol=0.0, atol=m * 1000.0 * np.finfo(dtype_).eps)
    assert u.dtype == dtype_
    assert cs.dtype == np.real(u).dtype
    assert vh.dtype == dtype_
    u, cs, vh = cossin([x[:p, :q], x[:p, q:], x[p:, :q], x[p:, q:]], swap_sign=swap_sign)
    assert_allclose(x, u @ cs @ vh, rtol=0.0, atol=m * 1000.0 * np.finfo(dtype_).eps)
    assert u.dtype == dtype_
    assert cs.dtype == np.real(u).dtype
    assert vh.dtype == dtype_
    _, cs2, vh2 = cossin(x, p, q, compute_u=False, swap_sign=swap_sign)
    assert_allclose(cs, cs2, rtol=0.0, atol=10 * np.finfo(dtype_).eps)
    assert_allclose(vh, vh2, rtol=0.0, atol=10 * np.finfo(dtype_).eps)
    u2, cs2, _ = cossin(x, p, q, compute_vh=False, swap_sign=swap_sign)
    assert_allclose(u, u2, rtol=0.0, atol=10 * np.finfo(dtype_).eps)
    assert_allclose(cs, cs2, rtol=0.0, atol=10 * np.finfo(dtype_).eps)
    _, cs2, _ = cossin(x, p, q, compute_u=False, compute_vh=False, swap_sign=swap_sign)
    assert_allclose(cs, cs2, rtol=0.0, atol=10 * np.finfo(dtype_).eps)