import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_constant_functions(self):

    @jit(nopython=True)
    def a():
        return 123

    @jit(nopython=True)
    def b():
        return 456

    @jit(nopython=True)
    def foo():
        return a() + b()
    r = foo()
    if r != 123 + 456:
        print(foo.overloads[()].library.get_llvm_str())
    self.assertEqual(r, 123 + 456)