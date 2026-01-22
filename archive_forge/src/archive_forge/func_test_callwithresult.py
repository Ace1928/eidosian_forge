import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_callwithresult(self):

    def process_result(result):
        return result * 2
    self._dll.tf_i.restype = process_result
    self._dll.tf_i.argtypes = (c_int,)
    self.assertEqual(self._dll.tf_i(42), 28)
    self.assertEqual(self.S(), 42)
    self.assertEqual(self._dll.tf_i(-42), -28)
    self.assertEqual(self.S(), -42)