import numpy as np
import ctypes
from numba import jit, literal_unroll, njit, typeof
from numba.core import types
from numba.core.itanium_mangler import mangle_type
from numba.core.errors import TypingError
import unittest
from numba.np import numpy_support
from numba.tests.support import TestCase, skip_ppc64le_issue6465
def test_from_dtype(self):
    rec = numpy_support.from_dtype(recordtype)
    self.assertEqual(rec.typeof('a'), types.float64)
    self.assertEqual(rec.typeof('b'), types.int16)
    self.assertEqual(rec.typeof('c'), types.complex64)
    self.assertEqual(rec.typeof('d'), types.UnicodeCharSeq(5))
    self.assertEqual(rec.offset('a'), recordtype.fields['a'][1])
    self.assertEqual(rec.offset('b'), recordtype.fields['b'][1])
    self.assertEqual(rec.offset('c'), recordtype.fields['c'][1])
    self.assertEqual(rec.offset('d'), recordtype.fields['d'][1])
    self.assertEqual(recordtype.itemsize, rec.size)