import collections
import itertools
import numpy as np
from numba import njit, jit, typeof, literally
from numba.core import types, errors, utils
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_array_tuple(self):
    aryty = types.Array(types.float64, 1, 'C')
    cfunc = njit((aryty, aryty))(tuple_return_usecase)
    a = b = np.arange(5, dtype='float64')
    ra, rb = cfunc(a, b)
    self.assertPreciseEqual(ra, a)
    self.assertPreciseEqual(rb, b)
    del a, b
    self.assertPreciseEqual(ra, rb)