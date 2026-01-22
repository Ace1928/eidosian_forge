import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_two_scalars(self):

    class Ref(ctypes.Structure):
        _fields_ = [('apple', ctypes.c_int32), ('orange', ctypes.c_float)]
    ty = types.Record.make_c_struct([('apple', types.int32), ('orange', types.float32)])
    self.assertEqual(len(ty), 2)
    self.assertEqual(ty.offset('apple'), Ref.apple.offset)
    self.assertEqual(ty.offset('orange'), Ref.orange.offset)
    self.assertEqual(ty.size, ctypes.sizeof(Ref))
    dtype = ty.dtype
    self.assertTrue(dtype.isalignedstruct)