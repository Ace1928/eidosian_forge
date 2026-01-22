import unittest
from ctypes import *
from ctypes.test import need_symbol
def test_del_segfault(self):
    BUF = c_char * 4
    buf = BUF()
    with self.assertRaises(AttributeError):
        del buf.raw