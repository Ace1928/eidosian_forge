import contextlib
import gc
from itertools import product, cycle
import sys
import warnings
from numbers import Number, Integral
import platform
import numpy as np
from numba import jit, njit, typeof
from numba.core import errors
from numba.tests.support import (TestCase, tag, needs_lapack, needs_blas,
from .matmul_usecase import matmul_usecase
import unittest
@needs_lapack
def test_no_input_mutation(self):
    X = np.array([[1.0, 3, 2, 7], [-5, 4, 2, 3], [9, -3, 1, 1], [2, -2, 2, 8]], order='F')
    X_orig = np.copy(X)

    @jit(nopython=True)
    def func(X, test):
        if test:
            X = X[1:2, :]
        return np.linalg.matrix_rank(X)
    expected = func.py_func(X, False)
    np.testing.assert_allclose(X, X_orig)
    got = func(X, False)
    np.testing.assert_allclose(X, X_orig)
    np.testing.assert_allclose(expected, got)