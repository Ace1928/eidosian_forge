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
def test_array_sliced_view(self):
    """
        Test .view() on A layout array but has contiguous innermost dimension.
        """
    pyfunc = array_sliced_view
    cfunc = njit((types.uint8[:],))(pyfunc)
    orig = np.array([1.5, 2], dtype=np.float32)
    byteary = orig.view(np.uint8)
    expect = pyfunc(byteary)
    got = cfunc(byteary)
    self.assertEqual(expect, got)