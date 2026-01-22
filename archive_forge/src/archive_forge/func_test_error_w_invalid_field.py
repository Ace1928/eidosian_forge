import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_error_w_invalid_field(self):
    arr = np.array([1, 2], dtype=recordtype3)
    jitfunc = njit(set_field1)
    with self.assertRaises(TypingError) as raises:
        jitfunc(arr[0])
    self.assertIn("Field 'f' was not found in record with fields ('first', 'second')", str(raises.exception))