import itertools
import pickle
import textwrap
import numpy as np
from numba import njit, vectorize
from numba.tests.support import MemoryLeakMixin, TestCase
from numba.core.errors import TypingError
import unittest
from numba.np.ufunc import dufunc
def test_dufunc_negative_axis(self):
    duadd = vectorize('int64(int64, int64)', identity=0)(pyuadd)

    @njit
    def foo(a, axis):
        return duadd.reduce(a, axis=axis)
    a = np.arange(40).reshape(5, 4, 2)
    cases = (0, -1, (0, -1), (-1, -2), (1, -1), -3)
    for axis in cases:
        expected = duadd.reduce(a, axis)
        got = foo(a, axis)
        self.assertPreciseEqual(expected, got)