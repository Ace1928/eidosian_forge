import sys
from functools import reduce
from numpy.testing import (assert_equal, assert_array_almost_equal, assert_,
import pytest
from pytest import raises as assert_raises
import numpy as np
from numpy import (eye, ones, zeros, zeros_like, triu, tril, tril_indices,
from numpy.random import rand, randint, seed
from scipy.linalg import (_flapack as flapack, lapack, inv, svd, cholesky,
from scipy.linalg.lapack import _compute_lwork
from scipy.stats import ortho_group, unitary_group
import scipy.sparse as sps
from scipy.linalg.lapack import get_lapack_funcs
from scipy.linalg.blas import get_blas_funcs
@pytest.mark.parametrize('dtype', DTYPES)
def test_gttrf_gttrs(dtype):
    seed(42)
    n = 10
    atol = 100 * np.finfo(dtype).eps
    du = generate_random_dtype_array((n - 1,), dtype=dtype)
    d = generate_random_dtype_array((n,), dtype=dtype)
    dl = generate_random_dtype_array((n - 1,), dtype=dtype)
    diag_cpy = [dl.copy(), d.copy(), du.copy()]
    A = np.diag(d) + np.diag(dl, -1) + np.diag(du, 1)
    x = np.random.rand(n)
    b = A @ x
    gttrf, gttrs = get_lapack_funcs(('gttrf', 'gttrs'), dtype=dtype)
    _dl, _d, _du, du2, ipiv, info = gttrf(dl, d, du)
    assert_array_equal(dl, diag_cpy[0])
    assert_array_equal(d, diag_cpy[1])
    assert_array_equal(du, diag_cpy[2])
    U = np.diag(_d, 0) + np.diag(_du, 1) + np.diag(du2, 2)
    L = np.eye(n, dtype=dtype)
    for i, m in enumerate(_dl):
        piv = ipiv[i] - 1
        L[:, [i, piv]] = L[:, [piv, i]]
        L[:, i] += L[:, i + 1] * m
    i, piv = (-1, ipiv[-1] - 1)
    L[:, [i, piv]] = L[:, [piv, i]]
    assert_allclose(A, L @ U, atol=atol)
    b_cpy = b.copy()
    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b)
    assert_array_equal(b, b_cpy)
    assert_allclose(x, x_gttrs, atol=atol)
    if dtype in REAL_DTYPES:
        trans = 'T'
        b_trans = A.T @ x
    else:
        trans = 'C'
        b_trans = A.conj().T @ x
    x_gttrs, info = gttrs(_dl, _d, _du, du2, ipiv, b_trans, trans=trans)
    assert_allclose(x, x_gttrs, atol=atol)
    with assert_raises(ValueError):
        gttrf(dl[:-1], d, du)
    with assert_raises(ValueError):
        gttrf(dl, d[:-1], du)
    with assert_raises(ValueError):
        gttrf(dl, d, du[:-1])
    with assert_raises(Exception):
        gttrf(dl[0], d[:1], du[0])
    du[0] = 0
    d[0] = 0
    __dl, __d, __du, _du2, _ipiv, _info = gttrf(dl, d, du)
    np.testing.assert_(__d[info - 1] == 0, '?gttrf: _d[info-1] is {}, not the illegal value :0.'.format(__d[info - 1]))