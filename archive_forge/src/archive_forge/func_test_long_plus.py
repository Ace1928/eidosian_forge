import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_long_plus(self):
    self._dll.tf_bl.restype = c_long
    self._dll.tf_bl.argtypes = (c_byte, c_long)
    self.assertEqual(self._dll.tf_bl(0, -2147483646), -715827882)
    self.assertEqual(self.S(), -2147483646)