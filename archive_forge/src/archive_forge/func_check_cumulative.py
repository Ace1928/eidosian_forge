from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def check_cumulative(self, pyfunc):
    arr = np.arange(2, 10, dtype=np.int16)
    expected, got = run_comparative(pyfunc, arr)
    self.assertPreciseEqual(got, expected)
    arr = np.linspace(2, 8, 6)
    expected, got = run_comparative(pyfunc, arr)
    self.assertPreciseEqual(got, expected)
    arr = arr.reshape((3, 2))
    expected, got = run_comparative(pyfunc, arr)
    self.assertPreciseEqual(got, expected)