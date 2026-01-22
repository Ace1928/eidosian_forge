import unittest
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.tests.support import TestCase, MemoryLeakMixin
from numba.np import numpy_support
def test_enumerate_refct(self):
    pyfunc = enumerate_array_usecase
    cfunc = njit(())(pyfunc)
    expected = pyfunc()
    self.assertPreciseEqual(cfunc(), expected)