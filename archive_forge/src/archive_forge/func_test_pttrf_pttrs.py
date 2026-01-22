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
@pytest.mark.parametrize('ddtype,dtype', zip(REAL_DTYPES + REAL_DTYPES, DTYPES))
def test_pttrf_pttrs(ddtype, dtype):
    seed(42)
    atol = 100 * np.finfo(dtype).eps
    n = 10
    d = generate_random_dtype_array((n,), ddtype) + 4
    e = generate_random_dtype_array((n - 1,), dtype)
    A = np.diag(d) + np.diag(e, -1) + np.diag(np.conj(e), 1)
    diag_cpy = [d.copy(), e.copy()]
    pttrf = get_lapack_funcs('pttrf', dtype=dtype)
    _d, _e, info = pttrf(d, e)
    assert_array_equal(d, diag_cpy[0])
    assert_array_equal(e, diag_cpy[1])
    assert_equal(info, 0, err_msg=f'pttrf: info = {info}, should be 0')
    L = np.diag(_e, -1) + np.diag(np.ones(n))
    D = np.diag(_d)
    assert_allclose(A, L @ D @ L.conjugate().T, atol=atol)
    x = generate_random_dtype_array((n,), dtype)
    b = A @ x
    pttrs = get_lapack_funcs('pttrs', dtype=dtype)
    _x, info = pttrs(_d, _e.conj(), b)
    assert_equal(info, 0, err_msg=f'pttrs: info = {info}, should be 0')
    assert_allclose(x, _x, atol=atol)