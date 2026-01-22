from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_nan_cumulative(self, pyfunc):
    cfunc = jit(nopython=True)(pyfunc)

    def check(a):
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)

    def _set_some_values_to_nan(a):
        p = a.size // 2
        np.put(a, np.random.choice(range(a.size), p, replace=False), np.nan)
        return a

    def a_variations():
        yield np.linspace(-1, 3, 60).reshape(3, 4, 5)
        yield np.array([np.inf, 3, 4])
        yield np.array([True, True, True, False])
        yield np.arange(1, 10)
        yield np.asfortranarray(np.arange(1, 64) - 33.3)
        yield np.arange(1, 10, dtype=np.float32)[::-1]
    for a in a_variations():
        check(a)
        check(_set_some_values_to_nan(a.astype(np.float64)))
    check(np.array([]))
    check(np.full(10, np.nan))
    parts = np.array([np.nan, 2, np.nan, 4, 5, 6, 7, 8, 9])
    a = parts + 1j * parts[::-1]
    a = a.reshape(3, 3)
    check(a)