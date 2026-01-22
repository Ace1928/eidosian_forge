import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_getitem_static_int_index(self):
    self._test_get_equal(getitem_0)
    self._test_get_equal(getitem_1)
    self._test_get_equal(getitem_2)
    rec = numpy_support.from_dtype(recordtype)
    with self.assertRaises(TypingError) as raises:
        self.get_cfunc(getitem_10, (rec[:], types.intp))
    msg = 'Requested index 10 is out of range'
    self.assertIn(msg, str(raises.exception))