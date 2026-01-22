import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_unique_dispatcher(self):

    def foo_template(funcs, x):
        r = x
        for f in funcs:
            r = f(r)
        return r
    a = jit(nopython=True)(lambda x: x + 1)
    b = jit(nopython=True)(lambda x: x + 2)
    foo = jit(nopython=True)(foo_template)
    a(0)
    a.disable_compile()
    r = foo((a, b), 0)
    self.assertEqual(r, 3)
    self.assertEqual(foo.signatures[0][0].dtype.is_precise(), True)