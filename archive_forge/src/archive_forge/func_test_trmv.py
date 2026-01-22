import math
import pytest
import numpy as np
from numpy.testing import (assert_equal, assert_almost_equal, assert_,
from pytest import raises as assert_raises
from numpy import float32, float64, complex64, complex128, arange, triu, \
from numpy.random import rand, seed
from scipy.linalg import _fblas as fblas, get_blas_funcs, toeplitz, solve
def test_trmv(self):
    seed(1234)
    for ind, dtype in enumerate(DTYPES):
        n = 3
        A = (rand(n, n) + eye(n)).astype(dtype)
        x = rand(3).astype(dtype)
        func, = get_blas_funcs(('trmv',), dtype=dtype)
        y1 = func(a=A, x=x)
        y2 = triu(A).dot(x)
        assert_array_almost_equal(y1, y2)
        y1 = func(a=A, x=x, diag=1)
        A[arange(n), arange(n)] = dtype(1)
        y2 = triu(A).dot(x)
        assert_array_almost_equal(y1, y2)
        y1 = func(a=A, x=x, diag=1, trans=1)
        y2 = triu(A).T.dot(x)
        assert_array_almost_equal(y1, y2)
        y1 = func(a=A, x=x, diag=1, trans=2)
        y2 = triu(A).conj().T.dot(x)
        assert_array_almost_equal(y1, y2)