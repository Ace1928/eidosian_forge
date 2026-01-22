import unittest
from ctypes import *
def test_int_callback(self):
    args = []

    def func(arg):
        args.append(arg)
        return arg
    cb = CFUNCTYPE(None, MyInt)(func)
    self.assertEqual(None, cb(42))
    self.assertEqual(type(args[-1]), MyInt)
    cb = CFUNCTYPE(c_int, c_int)(func)
    self.assertEqual(42, cb(42))
    self.assertEqual(type(args[-1]), int)