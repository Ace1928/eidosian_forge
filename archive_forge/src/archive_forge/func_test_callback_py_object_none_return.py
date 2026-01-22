import unittest
from test import support
import ctypes
import gc
import _ctypes_test
@support.refcount_test
def test_callback_py_object_none_return(self):
    for FUNCTYPE in (ctypes.CFUNCTYPE, ctypes.PYFUNCTYPE):
        with self.subTest(FUNCTYPE=FUNCTYPE):

            @FUNCTYPE(ctypes.py_object)
            def func():
                return None
            for _ in range(10000):
                func()