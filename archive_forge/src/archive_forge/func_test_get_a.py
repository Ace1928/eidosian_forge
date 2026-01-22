import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_get_a(self):
    self._test_get_equal(get_a)
    self._test_get_equal(get_a_subarray)
    self._test_get_equal(getitem_a)
    self._test_get_equal(getitem_a_subarray)
    self._test_get_equal(get_a_zero)
    self._test_get_equal(get_zero_a)