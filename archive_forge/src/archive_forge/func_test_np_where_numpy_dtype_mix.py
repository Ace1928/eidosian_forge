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
def test_np_where_numpy_dtype_mix(self):
    pyfunc = np_where_3
    cfunc = jit(nopython=True)(pyfunc)
    c = np.array([False, True, False, False, False, False, True, False, False, False, True, False])
    a = np.uint32(1)
    b = np.array([5.0, 0.0, 3.0, 2.0, -1.0, -4.0, 0.0, -10.0, 10.0, 1.0, 0.0, 3.0], dtype=np.float64)
    r = np.array([5.0, 1.0, 3.0, 2.0, -1.0, -4.0, 1.0, -10.0, 10.0, 1.0, 1.0, 3.0], dtype=np.float64)
    np.testing.assert_equal(cfunc(c, a, b), r)
    a = a.astype(np.float32)
    b = b.astype(np.int64)
    np.testing.assert_equal(cfunc(c, a, b), r)
    c = c.astype(int)
    c[c != 0] = 34242324
    np.testing.assert_equal(cfunc(c, a, b), r)
    tmpmask = c != 0
    c[c == 0] = 41247212
    c[tmpmask] = 0
    np.testing.assert_equal(cfunc(c, b, a), r)