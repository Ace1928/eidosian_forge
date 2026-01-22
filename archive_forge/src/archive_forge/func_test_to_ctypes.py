from ctypes import *
import sys
import threading
import numpy as np
from numba import jit, njit
from numba.core import types, errors
from numba.core.typing import ctypes_utils
from numba.tests.support import MemoryLeakMixin, tag, TestCase
from numba.tests.ctypes_usecases import *
import unittest
def test_to_ctypes(self):
    """
        Test converting a Numba type to a ctypes type.
        """

    def check(cty, ty):
        got = ctypes_utils.to_ctypes(ty)
        self.assertEqual(got, cty)
    self._conversion_tests(check)
    with self.assertRaises(TypeError) as raises:
        ctypes_utils.to_ctypes(types.ellipsis)
    self.assertIn("Cannot convert Numba type '...' to ctypes type", str(raises.exception))