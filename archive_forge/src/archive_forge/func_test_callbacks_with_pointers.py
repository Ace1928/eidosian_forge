import unittest, sys
from ctypes import *
import _ctypes_test
def test_callbacks_with_pointers(self):
    PROTOTYPE = CFUNCTYPE(c_int, POINTER(c_int))
    self.result = []

    def func(arg):
        for i in range(10):
            self.result.append(arg[i])
        return 0
    callback = PROTOTYPE(func)
    dll = CDLL(_ctypes_test.__file__)
    doit = dll._testfunc_callback_with_pointer
    doit(callback)
    doit(callback)