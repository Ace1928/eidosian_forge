import functools
import unittest
from test import support
from ctypes import *
from ctypes.test import need_symbol
from _ctypes import CTYPES_MAX_ARGCOUNT
import _ctypes_test
def test_convert_result_error(self):

    def func():
        return ('tuple',)
    proto = CFUNCTYPE(c_int)
    ctypes_func = proto(func)
    with support.catch_unraisable_exception() as cm:
        result = ctypes_func()
        self.assertIsInstance(cm.unraisable.exc_value, TypeError)
        self.assertEqual(cm.unraisable.err_msg, 'Exception ignored on converting result of ctypes callback function')
        self.assertIs(cm.unraisable.object, func)