from ctypes import *
from ctypes.test import need_symbol
import unittest
import _ctypes_test
def test_paramflags(self):
    prototype = CFUNCTYPE(c_void_p, c_void_p)
    func = prototype(('_testfunc_p_p', testdll), ((1, 'input'),))
    try:
        func()
    except TypeError as details:
        self.assertEqual(str(details), "required argument 'input' missing")
    else:
        self.fail('TypeError not raised')
    self.assertEqual(func(None), None)
    self.assertEqual(func(input=None), None)