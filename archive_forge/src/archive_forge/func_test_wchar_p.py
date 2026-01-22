from ctypes import *
from ctypes.test import need_symbol
import unittest
import sys
@need_symbol('c_wchar_p')
def test_wchar_p(self):
    s = c_wchar_p('hiho')
    self.assertEqual(cast(cast(s, c_void_p), c_wchar_p).value, 'hiho')