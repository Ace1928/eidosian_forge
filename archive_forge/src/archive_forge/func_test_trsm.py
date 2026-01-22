import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_trsm():
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        tol = np.finfo(dtype).eps * 1000
        func, = get_blas_funcs(('trsm',), dtype=dtype)
        A = rand(4, 5).astype(dtype)
        B = rand(4, 4).astype(dtype)
        alpha = dtype(1)
        assert_raises(Exception, func, alpha, A, B)
        assert_raises(Exception, func, alpha, A.T, B)
        n = 8
        m = 7
        alpha = dtype(-2.5)
        A = (rand(m, m) if ind < 2 else rand(m, m) + rand(m, m) * 1j) + eye(m)
        A = A.astype(dtype)
        Au = triu(A)
        Al = tril(A)
        B1 = rand(m, n).astype(dtype)
        B2 = rand(n, m).astype(dtype)
        x1 = func(alpha=alpha, a=A, b=B1)
        assert_equal(B1.shape, x1.shape)
        x2 = solve(Au, alpha * B1)
        assert_allclose(x1, x2, atol=tol)
        x1 = func(alpha=alpha, a=A, b=B1, trans_a=1)
        x2 = solve(Au.T, alpha * B1)
        assert_allclose(x1, x2, atol=tol)
        x1 = func(alpha=alpha, a=A, b=B1, trans_a=2)
        x2 = solve(Au.conj().T, alpha * B1)
        assert_allclose(x1, x2, atol=tol)
        x1 = func(alpha=alpha, a=A, b=B1, diag=1)
        Au[arange(m), arange(m)] = dtype(1)
        x2 = solve(Au, alpha * B1)
        assert_allclose(x1, x2, atol=tol)
        x1 = func(alpha=alpha, a=A, b=B2, diag=1, side=1)
        x2 = solve(Au.conj().T, alpha * B2.conj().T)
        assert_allclose(x1, x2.conj().T, atol=tol)
        x1 = func(alpha=alpha, a=A, b=B2, diag=1, side=1, lower=1)
        Al[arange(m), arange(m)] = dtype(1)
        x2 = solve(Al.conj().T, alpha * B2.conj().T)
        assert_allclose(x1, x2.conj().T, atol=tol)