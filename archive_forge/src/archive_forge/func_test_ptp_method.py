from itertools import product, combinations_with_replacement
import numpy as np
from numba import jit, njit, typeof
from numba.np.numpy_support import numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, tag
import unittest
def test_ptp_method(self):
    pyfunc = array_ptp
    cfunc = jit(nopython=True)(pyfunc)
    a = np.arange(10)
    expected = pyfunc(a)
    got = cfunc(a)
    self.assertPreciseEqual(expected, got)