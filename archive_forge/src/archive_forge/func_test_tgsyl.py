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
@pytest.mark.parametrize('dtype', REAL_DTYPES)
@pytest.mark.parametrize('trans', ('N', 'T'))
@pytest.mark.parametrize('ijob', [0, 1, 2, 3, 4])
def test_tgsyl(dtype, trans, ijob):
    atol = 0.001 if dtype == np.float32 else 1e-10
    rng = np.random.default_rng(1685779866898198)
    m, n = (10, 15)
    a, d, *_ = qz(rng.uniform(-10, 10, [m, m]).astype(dtype), rng.uniform(-10, 10, [m, m]).astype(dtype), output='real')
    b, e, *_ = qz(rng.uniform(-10, 10, [n, n]).astype(dtype), rng.uniform(-10, 10, [n, n]).astype(dtype), output='real')
    c = rng.uniform(-2, 2, [m, n]).astype(dtype)
    f = rng.uniform(-2, 2, [m, n]).astype(dtype)
    tgsyl = get_lapack_funcs('tgsyl', dtype=dtype)
    rout, lout, scale, dif, info = tgsyl(a, b, c, d, e, f, trans=trans, ijob=ijob)
    assert info == 0, 'INFO is non-zero'
    assert scale >= 0.0, 'SCALE must be non-negative'
    if ijob == 0:
        assert_allclose(dif, 0.0, rtol=0, atol=np.finfo(dtype).eps * 100, err_msg='DIF must be 0 for ijob =0')
    else:
        assert dif >= 0.0, 'DIF must be non-negative'
    if ijob <= 2:
        if trans == 'N':
            lhs1 = a @ rout - lout @ b
            rhs1 = scale * c
            lhs2 = d @ rout - lout @ e
            rhs2 = scale * f
        elif trans == 'T':
            lhs1 = np.transpose(a) @ rout + np.transpose(d) @ lout
            rhs1 = scale * c
            lhs2 = rout @ np.transpose(b) + lout @ np.transpose(e)
            rhs2 = -1.0 * scale * f
        assert_allclose(lhs1, rhs1, atol=atol, rtol=0.0, err_msg='lhs1 and rhs1 do not match')
        assert_allclose(lhs2, rhs2, atol=atol, rtol=0.0, err_msg='lhs2 and rhs2 do not match')