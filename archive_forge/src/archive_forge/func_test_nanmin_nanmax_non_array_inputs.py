from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_nanmin_nanmax_non_array_inputs(self):
    pyfuncs = (array_nanmin, array_nanmax)

    def check(a):
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)

    def a_variations():
        yield [1, 6, 4, 2]
        yield ((-10, 4, -12), (5, 200, -30))
        yield np.array(3)
        yield (2,)
        yield 3.142
        yield False
        yield (np.nan, 3.142, -5.2, 3.0)
        yield [np.inf, np.nan, -np.inf]
        yield [(np.nan, 1.1), (-4.4, 8.7)]
    for pyfunc in pyfuncs:
        cfunc = jit(nopython=True)(pyfunc)
        for a in a_variations():
            check(a)