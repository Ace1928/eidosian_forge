import numpy as np
import unittest
from numba.np.numpy_support import from_dtype
from numba import njit, typeof
from numba.core import types
from numba.tests.support import (TestCase, MemoryLeakMixin,
from numba.core.errors import TypingError
from numba.experimental import jitclass
def test_use_dtype(self):
    b = np.empty(1, dtype=np.int16)
    pyfunc = use_dtype
    cfunc = self.get_cfunc(pyfunc, (typeof(self.a), typeof(b)))
    expected = pyfunc(self.a, b)
    self.assertPreciseEqual(cfunc(self.a, b), expected)