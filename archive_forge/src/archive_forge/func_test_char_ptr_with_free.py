import unittest
from ctypes import *
from ctypes.test import need_symbol
import _ctypes_test
def test_char_ptr_with_free(self):
    dll = CDLL(_ctypes_test.__file__)
    s = b'abcdefghijklmnopqrstuvwxyz'

    class allocated_c_char_p(c_char_p):
        pass
    dll.my_free.restype = None

    def errcheck(result, func, args):
        retval = result.value
        dll.my_free(result)
        return retval
    dll.my_strdup.restype = allocated_c_char_p
    dll.my_strdup.errcheck = errcheck
    try:
        res = dll.my_strdup(s)
        self.assertEqual(res, s)
    finally:
        del dll.my_strdup.errcheck