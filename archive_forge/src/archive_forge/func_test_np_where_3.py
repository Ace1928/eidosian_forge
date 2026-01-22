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
def test_np_where_3(self):
    pyfunc = np_where_3

    def fac(N):
        np.random.seed(42)
        arr = np.random.random(N)
        arr[arr < 0.3] = 0.0
        arr[arr > 0.7] = float('nan')
        return arr
    layouts = cycle(['C', 'F', 'A'])
    _types = [np.int32, np.int64, np.float32, np.float64, np.complex64, np.complex128]
    np.random.seed(42)

    def check_arr(arr, layout=False):
        np.random.shuffle(_types)
        if layout != False:
            x = np.zeros_like(arr, dtype=_types[0], order=layout)
            y = np.zeros_like(arr, dtype=_types[1], order=layout)
            arr = arr.copy(order=layout)
        else:
            x = np.zeros_like(arr, dtype=_types[0], order=next(layouts))
            y = np.zeros_like(arr, dtype=_types[1], order=next(layouts))
        x.fill(4)
        y.fill(9)
        cfunc = njit((typeof(arr), typeof(x), typeof(y)))(pyfunc)
        expected = pyfunc(arr, x, y)
        got = cfunc(arr, x, y)
        self.assertPreciseEqual(got, expected)

    def check_scal(scal):
        x = 4
        y = 5
        np.random.shuffle(_types)
        x = _types[0](4)
        y = _types[1](5)
        cfunc = njit((typeof(scal), typeof(x), typeof(y)))(pyfunc)
        expected = pyfunc(scal, x, y)
        got = cfunc(scal, x, y)
        self.assertPreciseEqual(got, expected)
    arr = np.int16([1, 0, -1, 0])
    check_arr(arr)
    arr = np.bool_([1, 0, 1])
    check_arr(arr)
    arr = fac(24)
    check_arr(arr)
    check_arr(arr.reshape((3, 8)))
    check_arr(arr.reshape((3, 8)).T)
    check_arr(arr.reshape((3, 8))[::2])
    check_arr(arr.reshape((2, 3, 4)))
    check_arr(arr.reshape((2, 3, 4)).T)
    check_arr(arr.reshape((2, 3, 4))[::2])
    check_arr(arr.reshape((2, 3, 4)), layout='F')
    check_arr(arr.reshape((2, 3, 4)).T, layout='F')
    check_arr(arr.reshape((2, 3, 4))[::2], layout='F')
    for v in (0.0, 1.5, float('nan')):
        arr = np.array([v]).reshape(())
        check_arr(arr)
    for x in (0, 1, True, False, 2.5, 0j):
        check_scal(x)