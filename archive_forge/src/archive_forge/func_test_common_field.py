import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_common_field(self):
    njit_sig = njit(types.float64(typeof(self.a_rec1)))
    functions = [njit(self.func), njit_sig(self.func)]
    for fc in functions:
        fc(self.a_rec1)
        fc.disable_compile()
        y = fc(self.ab_rec1)
        self.assertEqual(self.value, y)