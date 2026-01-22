import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_tuple_of_arrays_iter(self):

    def bar(arrs):
        total = 0
        for arr in arrs:
            total += arr[0]
        return total
    x = y = np.arange(3, dtype=np.int32)
    aryty = types.Array(types.int32, 1, 'C')
    cfunc = njit((types.containers.UniTuple(aryty, 2),))(bar)
    expect = bar((x, y))
    got = cfunc((x, y))
    self.assertEqual(expect, got)