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
@pytest.mark.parametrize('lower', [False, True])
@pytest.mark.parametrize('dtype', DTYPES)
def test_pptrs_pptri_pptrf_ppsv_ppcon(dtype, lower):
    seed(1234)
    atol = np.finfo(dtype).eps * 100
    n, nrhs = (10, 4)
    a = generate_random_dtype_array([n, n], dtype=dtype)
    b = generate_random_dtype_array([n, nrhs], dtype=dtype)
    a = a.conj().T + a + np.eye(n, dtype=dtype) * dtype(5.0)
    if lower:
        inds = ([x for y in range(n) for x in range(y, n)], [y for y in range(n) for x in range(y, n)])
    else:
        inds = ([x for y in range(1, n + 1) for x in range(y)], [y - 1 for y in range(1, n + 1) for x in range(y)])
    ap = a[inds]
    ppsv, pptrf, pptrs, pptri, ppcon = get_lapack_funcs(('ppsv', 'pptrf', 'pptrs', 'pptri', 'ppcon'), dtype=dtype, ilp64='preferred')
    ul, info = pptrf(n, ap, lower=lower)
    assert_equal(info, 0)
    aul = cholesky(a, lower=lower)[inds]
    assert_allclose(ul, aul, rtol=0, atol=atol)
    uli, info = pptri(n, ul, lower=lower)
    assert_equal(info, 0)
    auli = inv(a)[inds]
    assert_allclose(uli, auli, rtol=0, atol=atol)
    x, info = pptrs(n, ul, b, lower=lower)
    assert_equal(info, 0)
    bx = solve(a, b)
    assert_allclose(x, bx, rtol=0, atol=atol)
    xv, info = ppsv(n, ap, b, lower=lower)
    assert_equal(info, 0)
    assert_allclose(xv, bx, rtol=0, atol=atol)
    anorm = np.linalg.norm(a, 1)
    rcond, info = ppcon(n, ap, anorm=anorm, lower=lower)
    assert_equal(info, 0)
    assert_(abs(1 / rcond - np.linalg.cond(a, p=1)) * rcond < 1)