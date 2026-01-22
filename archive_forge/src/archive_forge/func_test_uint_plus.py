import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_uint_plus(self):
    self._dll.tf_bI.restype = c_uint
    self._dll.tf_bI.argtypes = (c_byte, c_uint)
    self.assertEqual(self._dll.tf_bI(0, 4294967295), 1431655765)
    self.assertEqual(self.U(), 4294967295)