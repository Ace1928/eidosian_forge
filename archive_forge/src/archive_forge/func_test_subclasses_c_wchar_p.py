import unittest
from ctypes.test import need_symbol
import test.support
@need_symbol('c_wchar_p')
def test_subclasses_c_wchar_p(self):
    from ctypes import c_wchar_p

    class CWCHARP(c_wchar_p):

        def from_param(cls, value):
            return value * 3
        from_param = classmethod(from_param)
    self.assertEqual(CWCHARP.from_param('abc'), 'abcabcabc')