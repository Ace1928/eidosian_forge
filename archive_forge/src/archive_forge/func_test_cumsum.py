from itertools import product, cycle
import gc
import sys
import unittest
import warnings
import numpy as np
from numba import jit, njit, typeof
from numba.core import types
from numba.core.errors import TypingError, NumbaValueError
from numba.np.numpy_support import as_dtype, numpy_version
from numba.tests.support import TestCase, MemoryLeakMixin, needs_blas
def test_cumsum(self):
    pyfunc = array_cumsum
    cfunc = jit(nopython=True)(pyfunc)
    a = np.ones((2, 3))
    self.assertPreciseEqual(pyfunc(a), cfunc(a))
    with self.assertRaises(TypingError):
        cfunc(a, 1)
    pyfunc = array_cumsum_kws
    cfunc = jit(nopython=True)(pyfunc)
    with self.assertRaises(TypingError):
        cfunc(a, axis=1)