import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_pr4967_array(self):
    import numpy as np

    @cfunc('intp(intp[:], float64[:])')
    def foo1(x, y):
        return x[0] + y[0]

    @cfunc('intp(intp[:], float64[:])')
    def foo2(x, y):
        return x[0] - y[0]

    def bar(fx, fy, i):
        a = np.array([10], dtype=np.intp)
        b = np.array([12], dtype=np.float64)
        if i == 0:
            f = fx
        elif i == 1:
            f = fy
        else:
            return
        return f(a, b)
    r = jit(nopython=True, no_cfunc_wrapper=True)(bar)(foo1, foo2, 0)
    self.assertEqual(r, bar(foo1, foo2, 0))
    self.assertNotEqual(r, bar(foo1, foo2, 1))