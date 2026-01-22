import unittest
from ctypes import *
def test_ignore_retval(self):
    proto = CFUNCTYPE(None)

    def func():
        return (1, 'abc', None)
    cb = proto(func)
    self.assertEqual(None, cb())