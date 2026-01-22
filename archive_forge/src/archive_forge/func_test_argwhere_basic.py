from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_argwhere_basic(self):
    pyfunc = numpy_argwhere
    cfunc = jit(nopython=True)(pyfunc)

    def a_variations():
        yield (np.arange(-5, 5) > 2)
        yield np.full(5, fill_value=0)
        yield np.full(5, fill_value=1)
        yield np.array([])
        yield np.array([-1.0, 0.0, 1.0])
        a = self.random.randn(100)
        yield (a > 0.2)
        yield (a.reshape(5, 5, 4) > 0.5)
        yield (a.reshape(50, 2, order='F') > 0.5)
        yield (a.reshape(25, 4)[1::2] > 0.5)
        yield (a == a - 1)
        yield (a > -a)
    for a in a_variations():
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)