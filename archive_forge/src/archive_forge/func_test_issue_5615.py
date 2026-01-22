import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_issue_5615(self):

    @njit
    def foo1(x):
        return x + 1

    @njit
    def foo2(x):
        return x + 2

    @njit
    def bar(fcs):
        x = 0
        a = 10
        i, j = fcs[0]
        x += i(j(a))
        for t in literal_unroll(fcs):
            i, j = t
            x += i(j(a))
        return x
    tup = ((foo1, foo2), (foo2, foo1))
    self.assertEqual(bar(tup), 39)