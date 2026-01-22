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
def test_clip_no_broadcast(self):
    self.disable_leak_check()
    cfunc = jit(nopython=True)(np_clip)
    msg = '.*shape mismatch: objects cannot be broadcast to a single shape.*'
    a = np.linspace(-10, 10, 40).reshape(5, 2, 4)
    a_min_arr = np.arange(-5, 0).astype(a.dtype).reshape(5, 1)
    a_max_arr = np.arange(0, 5).astype(a.dtype).reshape(5, 1)
    min_max = [(0, a_max_arr), (-5, a_max_arr), (a_min_arr, a_max_arr), (a_min_arr, 0), (a_min_arr, 5)]
    for a_min, a_max in min_max:
        with self.assertRaisesRegex(ValueError, msg):
            cfunc(a, a_min, a_max)