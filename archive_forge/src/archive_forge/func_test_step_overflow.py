import unittest
from test.support import bigmemtest, _2G
import sys
from ctypes import *
from ctypes.test import need_symbol
def test_step_overflow(self):
    a = (c_int * 5)()
    a[3::sys.maxsize] = (1,)
    self.assertListEqual(a[3::sys.maxsize], [1])
    a = (c_char * 5)()
    a[3::sys.maxsize] = b'A'
    self.assertEqual(a[3::sys.maxsize], b'A')
    a = (c_wchar * 5)()
    a[3::sys.maxsize] = u'X'
    self.assertEqual(a[3::sys.maxsize], u'X')