from ctypes import *
import unittest, sys
from test import support
import _ctypes_test
def test_HWND(self):
    from ctypes import wintypes
    self.assertEqual(sizeof(wintypes.HWND), sizeof(c_void_p))