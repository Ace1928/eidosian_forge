import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def test_number_real(self):
    """
        Testing .real of non-complex dtypes
        """
    for dtype in [np.uint8, np.int32, np.float32, np.float64]:
        self.check_number_real(dtype)