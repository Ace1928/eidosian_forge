from functools import partial
from itertools import permutations
import numpy as np
import unittest
from numba import jit, njit, from_dtype, typeof
from numba.core.errors import TypingError
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
def test_flatnonzero_array_like(self):
    pyfunc = numpy_flatnonzero
    cfunc = jit(nopython=True)(pyfunc)
    for a in self.array_like_variations():
        expected = pyfunc(a)
        got = cfunc(a)
        self.assertPreciseEqual(expected, got)