import unittest
from ctypes.test import need_symbol
import test.support
@test.support.cpython_only
def test_issue31311(self):
    from ctypes import Structure

    class BadStruct(Structure):

        @property
        def __dict__(self):
            pass
    with self.assertRaises(TypeError):
        BadStruct().__setstate__({}, b'foo')

    class WorseStruct(Structure):

        @property
        def __dict__(self):
            1 / 0
    with self.assertRaises(ZeroDivisionError):
        WorseStruct().__setstate__({}, b'foo')