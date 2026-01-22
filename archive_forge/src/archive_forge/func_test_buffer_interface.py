from ctypes import *
from ctypes.test import need_symbol
import unittest
def test_buffer_interface(self):
    self.assertEqual(len(bytearray(create_string_buffer(0))), 0)
    self.assertEqual(len(bytearray(create_string_buffer(1))), 1)