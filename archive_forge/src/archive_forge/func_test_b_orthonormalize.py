import itertools
import platform
import sys
import pytest
import numpy as np
from numpy import ones, r_, diag
from numpy.testing import (assert_almost_equal, assert_equal,
from scipy import sparse
from scipy.linalg import eig, eigh, toeplitz, orth
from scipy.sparse import spdiags, diags, eye, csr_matrix
from scipy.sparse.linalg import eigs, LinearOperator
from scipy.sparse.linalg._eigen.lobpcg import lobpcg
from scipy.sparse.linalg._eigen.lobpcg.lobpcg import _b_orthonormalize
from scipy._lib._util import np_long, np_ulong
@pytest.mark.parametrize('n', [50])
@pytest.mark.parametrize('m', [1, 2, 10])
@pytest.mark.parametrize('Vdtype', sorted(REAL_DTYPES, key=str))
@pytest.mark.parametrize('Bdtype', sorted(REAL_DTYPES, key=str))
@pytest.mark.parametrize('BVdtype', sorted(REAL_DTYPES, key=str))
def test_b_orthonormalize(n, m, Vdtype, Bdtype, BVdtype):
    """Test B-orthonormalization by Cholesky with callable 'B'.
    The function '_b_orthonormalize' is key in LOBPCG but may
    lead to numerical instabilities. The input vectors are often
    badly scaled, so the function needs scale-invariant Cholesky;
    see https://netlib.org/lapack/lawnspdf/lawn14.pdf.
    """
    rnd = np.random.RandomState(0)
    X = rnd.standard_normal((n, m)).astype(Vdtype)
    Xcopy = np.copy(X)
    vals = np.arange(1, n + 1, dtype=float)
    B = diags([vals], [0], (n, n)).astype(Bdtype)
    BX = B @ X
    BX = BX.astype(BVdtype)
    dtype = min(X.dtype, B.dtype, BX.dtype)
    atol = m * n * max(np.finfo(dtype).eps, np.finfo(np.float64).eps)
    Xo, BXo, _ = _b_orthonormalize(lambda v: B @ v, X, BX)
    assert_equal(X, Xo)
    assert_equal(id(X), id(Xo))
    assert_equal(BX, BXo)
    assert_equal(id(BX), id(BXo))
    assert_allclose(B @ Xo, BXo, atol=atol, rtol=atol)
    assert_allclose(Xo.T.conj() @ B @ Xo, np.identity(m), atol=atol, rtol=atol)
    X = np.copy(Xcopy)
    Xo1, BXo1, _ = _b_orthonormalize(lambda v: B @ v, X)
    assert_allclose(Xo, Xo1, atol=atol, rtol=atol)
    assert_allclose(BXo, BXo1, atol=atol, rtol=atol)
    assert_equal(X, Xo1)
    assert_equal(id(X), id(Xo1))
    assert_allclose(B @ Xo1, BXo1, atol=atol, rtol=atol)
    scaling = 1.0 / np.geomspace(10, 10000000000.0, num=m)
    X = Xcopy * scaling
    X = X.astype(Vdtype)
    BX = B @ X
    BX = BX.astype(BVdtype)
    Xo1, BXo1, _ = _b_orthonormalize(lambda v: B @ v, X, BX)
    Xo1 = sign_align(Xo1, Xo)
    assert_allclose(Xo, Xo1, atol=atol, rtol=atol)
    BXo1 = sign_align(BXo1, BXo)
    assert_allclose(BXo, BXo1, atol=atol, rtol=atol)