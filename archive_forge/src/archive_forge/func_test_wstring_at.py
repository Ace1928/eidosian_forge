import sys
from test import support
import unittest
from ctypes import *
from ctypes.test import need_symbol
@need_symbol('create_unicode_buffer')
def test_wstring_at(self):
    p = create_unicode_buffer('Hello, World')
    a = create_unicode_buffer(1000000)
    result = memmove(a, p, len(p) * sizeof(c_wchar))
    self.assertEqual(a.value, 'Hello, World')
    self.assertEqual(wstring_at(a), 'Hello, World')
    self.assertEqual(wstring_at(a, 5), 'Hello')
    self.assertEqual(wstring_at(a, 16), 'Hello, World\x00\x00\x00\x00')
    self.assertEqual(wstring_at(a, 0), '')