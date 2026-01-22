from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_fill(self):
    pyfunc = array_fill
    cfunc = jit(nopython=True)(pyfunc)

    def check(arr, val):
        expected = np.copy(arr)
        erv = pyfunc(expected, val)
        self.assertTrue(erv is None)
        got = np.copy(arr)
        grv = cfunc(got, val)
        self.assertTrue(grv is None)
        self.assertPreciseEqual(expected, got)
    A = np.arange(1)
    for x in [np.float64, np.bool_]:
        check(A, x(10))
    A = np.arange(12).reshape(3, 4)
    for x in [np.float64, np.bool_]:
        check(A, x(10))
    A = np.arange(48, dtype=np.complex64).reshape(2, 3, 4, 2)
    for x in [np.float64, np.complex128, np.bool_]:
        check(A, x(10))