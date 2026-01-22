import unittest, sys
from ctypes import *
import _ctypes_test
def test_pointers_bool(self):
    self.assertEqual(bool(POINTER(c_int)()), False)
    self.assertEqual(bool(pointer(c_int())), True)
    self.assertEqual(bool(CFUNCTYPE(None)(0)), False)
    self.assertEqual(bool(CFUNCTYPE(None)(42)), True)
    if sys.platform == 'win32':
        mth = WINFUNCTYPE(None)(42, 'name', (), None)
        self.assertEqual(bool(mth), True)