import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_tuple_iter_issue1504(self):

    def bar(x, y):
        total = 0
        for row in zip(x, y):
            total += row[0] + row[1]
        return total
    x = y = np.arange(3, dtype=np.int32)
    aryty = types.Array(types.int32, 1, 'C')
    cfunc = njit((aryty, aryty))(bar)
    expect = bar(x, y)
    got = cfunc(x, y)
    self.assertEqual(expect, got)