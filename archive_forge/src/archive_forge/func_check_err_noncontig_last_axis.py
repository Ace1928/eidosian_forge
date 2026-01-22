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
def check_err_noncontig_last_axis(arr, dtype):
    msg = 'To change to a dtype of a different size, the last axis must be contiguous'
    with self.assertRaises(ValueError) as raises:
        make_array_view(dtype)(arr)
    self.assertEqual(str(raises.exception), msg)
    with self.assertRaises(ValueError) as raises:
        run(arr, dtype)
    self.assertEqual(str(raises.exception), msg)