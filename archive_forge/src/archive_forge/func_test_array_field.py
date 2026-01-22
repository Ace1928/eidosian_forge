import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_array_field(self):
    rec1 = np.empty(1, dtype=[('a', 'f8', (4,))])[0]
    rec1['a'][0] = 1
    rec2 = np.empty(1, dtype=[('a', 'f8', (4,)), ('b', 'f8')])[0]
    rec2['a'][0] = self.value

    @njit
    def foo(rec):
        return rec['a'][0]
    foo(rec1)
    foo.disable_compile()
    y = foo(rec2)
    self.assertEqual(self.value, y)