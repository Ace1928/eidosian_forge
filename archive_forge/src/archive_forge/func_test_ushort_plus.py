import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_ushort_plus(self):
    self._dll.tf_bH.restype = c_ushort
    self._dll.tf_bH.argtypes = (c_byte, c_ushort)
    self.assertEqual(self._dll.tf_bH(0, 65535), 21845)
    self.assertEqual(self.U(), 65535)