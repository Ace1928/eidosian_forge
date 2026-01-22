import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def test_size_after_slicing(self):
    pyfunc = size_after_slicing_usecase
    cfunc = njit(pyfunc)
    arr = np.arange(2 * 5).reshape(2, 5)
    for i in range(arr.shape[0]):
        self.assertEqual(pyfunc(arr, i), cfunc(arr, i))
    arr = np.arange(2 * 5 * 3).reshape(2, 5, 3)
    for i in range(arr.shape[0]):
        self.assertEqual(pyfunc(arr, i), cfunc(arr, i))