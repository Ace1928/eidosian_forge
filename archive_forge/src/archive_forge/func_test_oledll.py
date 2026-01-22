import unittest
from ctypes import *
from ctypes.test import need_symbol
@need_symbol('oledll')
def test_oledll(self):
    self.assertRaises(OSError, oledll.oleaut32.CreateTypeLib2, 0, None, None)