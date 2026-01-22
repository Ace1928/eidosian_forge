import unittest
import types as pytypes
from numba import jit, njit, cfunc, types, int64, float64, float32, errors
from numba import literal_unroll, typeof
from numba.core.config import IS_WIN32
import ctypes
import warnings
from .support import TestCase, MemoryLeakMixin
import numpy as np
def test_generators(self):

    @jit(forceobj=True)
    def gen(xs):
        for x in xs:
            x += 1
            yield x

    @jit(forceobj=True)
    def con(gen_fn, xs):
        return [it for it in gen_fn(xs)]
    self.assertEqual(con(gen, (1, 2, 3)), [2, 3, 4])

    @jit(nopython=True)
    def gen_(xs):
        for x in xs:
            x += 1
            yield x
    self.assertEqual(con(gen_, (1, 2, 3)), [2, 3, 4])