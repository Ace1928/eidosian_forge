import numpy as np
import unittest
from numba import njit
from numba.core.errors import TypingError
from numba import jit, typeof
from numba.core import types
from numba.tests.support import TestCase
def test_constant_bytes(self):
    pyfunc = bytes_as_const_array
    cfunc = njit(())(pyfunc)
    np.testing.assert_array_equal(pyfunc(), cfunc())