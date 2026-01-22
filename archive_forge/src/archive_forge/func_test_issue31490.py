import unittest
import test.support
from ctypes import *
@test.support.cpython_only
def test_issue31490(self):
    with self.assertRaises(AttributeError):

        class Name(Structure):
            _fields_ = []
            _anonymous_ = ['x']
            x = 42