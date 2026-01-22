import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_apply_function_in_function(self):

    def foo(f, f_inner):
        return f(f_inner)

    @cfunc('int64(float64)')
    def f_inner(i):
        return int64(i * 3)

    @cfunc(int64(types.FunctionType(f_inner._sig)))
    def f(f_inner):
        return f_inner(123.4)
    self.assertEqual(jit(nopython=True)(foo)(f, f_inner), foo(f._pyfunc, f_inner._pyfunc))