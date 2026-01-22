import sys
from test import support
import unittest
from ctypes import *
from ctypes.test import need_symbol
@support.refcount_test
def test_string_at(self):
    s = string_at(b'foo bar')
    self.assertEqual(2, sys.getrefcount(s))
    self.assertTrue(s, 'foo bar')
    self.assertEqual(string_at(b'foo bar', 7), b'foo bar')
    self.assertEqual(string_at(b'foo bar', 3), b'foo')