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
def test_np_where_numpy_test_error(self):
    pyfunc = np_where_3
    cfunc = jit(nopython=True)(pyfunc)
    c = [True, True]
    a = np.ones((4, 5))
    b = np.ones((5, 5))
    self.disable_leak_check()
    with self.assertRaisesRegex(ValueError, 'objects cannot be broadcast'):
        cfunc(c, a, b)
    with self.assertRaisesRegex(ValueError, 'objects cannot be broadcast'):
        cfunc(c[0], a, b)