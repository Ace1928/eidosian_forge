import unittest
from ctypes import *
from sys import getrefcount as grc
def test_c_char_p(self):
    s = b'Hello, World'
    refcnt = grc(s)
    cs = c_char_p(s)
    self.assertEqual(refcnt + 1, grc(s))
    self.assertSame(cs._objects, s)