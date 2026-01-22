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
def check_layout_dependent_func(self, pyfunc, fac=np.arange):

    def is_same(a, b):
        return a.ctypes.data == b.ctypes.data

    def check_arr(arr):
        cfunc = njit((typeof(arr),))(pyfunc)
        expected = pyfunc(arr)
        got = cfunc(arr)
        self.assertPreciseEqual(expected, got)
        self.assertEqual(is_same(expected, arr), is_same(got, arr))
    arr = fac(24)
    check_arr(arr)
    check_arr(arr.reshape((3, 8)))
    check_arr(arr.reshape((3, 8)).T)
    check_arr(arr.reshape((3, 8))[::2])
    check_arr(arr.reshape((2, 3, 4)))
    check_arr(arr.reshape((2, 3, 4)).T)
    check_arr(arr.reshape((2, 3, 4))[::2])
    arr = np.array([0]).reshape(())
    check_arr(arr)