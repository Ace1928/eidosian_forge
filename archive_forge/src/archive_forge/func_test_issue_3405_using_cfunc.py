import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_issue_3405_using_cfunc(self):

    @cfunc('int64()')
    def a():
        return 2

    @cfunc('int64()')
    def b():
        return 3

    def g(arg):
        if arg:
            f = a
        else:
            f = b
        return f()
    self.assertEqual(jit(nopython=True)(g)(True), 2)
    self.assertEqual(jit(nopython=True)(g)(False), 3)