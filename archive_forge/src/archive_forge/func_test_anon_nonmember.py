import unittest
import test.support
from ctypes import *
def test_anon_nonmember(self):
    self.assertRaises(AttributeError, lambda: type(Structure)('Name', (Structure,), {'_fields_': [], '_anonymous_': ['x']}))