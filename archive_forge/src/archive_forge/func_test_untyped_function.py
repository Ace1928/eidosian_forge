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
def test_untyped_function(self):
    with self.assertRaises(TypeError) as raises:
        njit((types.double,))(use_c_untyped)
    self.assertIn("ctypes function '_numba_test_exp' doesn't define its argument types", str(raises.exception))