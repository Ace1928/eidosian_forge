import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_tpsv(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 10
        x = rand(n).astype(dtype)
        A = triu(rand(n, n)) if ind < 2 else triu(rand(n, n) + rand(n, n) * 1j)
        A += eye(n)
        c, r = tril_indices(n)
        Ap = A[r, c]
        func, = get_blas_funcs(('tpsv',), dtype=dtype)
        y1 = func(n=n, ap=Ap, x=x)
        y2 = solve(A, x)
        assert_array_almost_equal(y1, y2)
        y1 = func(n=n, ap=Ap, x=x, diag=1)
        A[arange(n), arange(n)] = dtype(1)
        y2 = solve(A, x)
        assert_array_almost_equal(y1, y2)
        y1 = func(n=n, ap=Ap, x=x, diag=1, trans=1)
        y2 = solve(A.T, x)
        assert_array_almost_equal(y1, y2)
        y1 = func(n=n, ap=Ap, x=x, diag=1, trans=2)
        y2 = solve(A.conj().T, x)
        assert_array_almost_equal(y1, y2)