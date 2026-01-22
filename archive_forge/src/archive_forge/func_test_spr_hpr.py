import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_spr_hpr(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES + COMPLEX_DTYPES):
        n = 3
        A = rand(n, n).astype(dtype)
        if ind > 1:
            A += rand(n, n) * 1j
        A = A.astype(dtype)
        A = A + A.T if ind < 4 else A + A.conj().T
        c, r = tril_indices(n)
        Ap = A[r, c]
        x = rand(n).astype(dtype)
        alpha = (DTYPES + COMPLEX_DTYPES)[mod(ind, 4)](2.5)
        if ind > 3:
            func, = get_blas_funcs(('hpr',), dtype=dtype)
            y2 = alpha * x[:, None].dot(x[None, :].conj()) + A
        else:
            func, = get_blas_funcs(('spr',), dtype=dtype)
            y2 = alpha * x[:, None].dot(x[None, :]) + A
        y1 = func(n=n, alpha=alpha, ap=Ap, x=x)
        y1f = zeros((3, 3), dtype=dtype)
        y1f[r, c] = y1
        y1f[c, r] = y1.conj() if ind > 3 else y1
        assert_array_almost_equal(y1f, y2)