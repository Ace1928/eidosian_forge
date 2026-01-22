import ctypes
import struct
from numba.tests.support import TestCase
from numba import _helperlib
def test_append_get_string(self):
    l = List(self, 8, 1)
    l.append(b'abcdefgh')
    self.assertEqual(len(l), 1)
    r = l[0]
    self.assertEqual(r, b'abcdefgh')