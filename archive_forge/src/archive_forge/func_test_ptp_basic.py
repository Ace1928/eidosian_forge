from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_ptp_basic(self):
    pyfunc = array_ptp_global
    cfunc = jit(nopython=True)(pyfunc)

    def check(a):
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)

    def a_variations():
        yield np.arange(10)
        yield np.array([-1.1, np.nan, 2.2])
        yield np.array([-np.inf, 5])
        yield (4, 2, 5)
        yield (1,)
        yield np.full(5, 5)
        yield [2.2, -2.3, 0.1]
        a = np.linspace(-10, 10, 16).reshape(4, 2, 2)
        yield a
        yield np.asfortranarray(a)
        yield a[::-1]
        np.random.RandomState(0).shuffle(a)
        yield a
        yield 6
        yield 6.5
        yield (-np.inf)
        yield (1 + 4j)
        yield [2.2, np.nan]
        yield [2.2, np.inf]
        yield ((4.1, 2.0, -7.6), (4.3, 2.7, 5.2))
        yield np.full(5, np.nan)
        yield (1 + np.nan * 1j)
        yield (np.nan + np.nan * 1j)
        yield np.nan
    for a in a_variations():
        check(a)