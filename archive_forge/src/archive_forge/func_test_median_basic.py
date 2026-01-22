from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_median_basic(self):
    pyfunc = array_median_global

    def variations(a):
        yield a
        a = a[::-1].copy()
        yield a
        np.random.shuffle(a)
        yield a
        a[a % 4 >= 1] = 3.5
        yield a
    self.check_median_basic(pyfunc, variations)