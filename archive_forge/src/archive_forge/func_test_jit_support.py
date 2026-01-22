import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_jit_support(self):

    @jit(nopython=True)
    def foo(f, x):
        return f(x)

    @jit()
    def a(x):
        return x + 1

    @jit()
    def a2(x):
        return x - 1

    @jit()
    def b(x):
        return x + 1.5
    self.assertEqual(foo(a, 1), 2)
    a2(5)
    self.assertEqual(foo(a2, 2), 1)
    self.assertEqual(foo(a2, 3), 2)
    self.assertEqual(foo(a, 2), 3)
    self.assertEqual(foo(a, 1.5), 2.5)
    self.assertEqual(foo(a2, 1), 0)
    self.assertEqual(foo(a, 2.5), 3.5)
    self.assertEqual(foo(b, 1.5), 3.0)
    self.assertEqual(foo(b, 1), 2.5)