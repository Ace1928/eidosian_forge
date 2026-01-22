from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_all_basic(self, pyfunc=array_all):
    cfunc = jit(nopython=True)(pyfunc)

    def check(arr):
        self.assertPreciseEqual(pyfunc(arr), cfunc(arr))
    arr = np.float64([1.0, 0.0, float('inf'), float('nan')])
    check(arr)
    arr[1] = -0.0
    check(arr)
    arr[1] = 1.5
    check(arr)
    arr = arr.reshape((2, 2))
    check(arr)
    check(arr[::-1])