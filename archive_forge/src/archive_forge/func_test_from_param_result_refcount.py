import unittest
from ctypes.test import need_symbol
import test.support
@test.support.cpython_only
def test_from_param_result_refcount(self):
    import _ctypes_test
    from ctypes import PyDLL, c_int, c_void_p, py_object, Structure

    class X(Structure):
        """This struct size is <= sizeof(void*)."""
        _fields_ = [('a', c_void_p)]

        def __del__(self):
            trace.append(4)

        @classmethod
        def from_param(cls, value):
            trace.append(2)
            return cls()
    PyList_Append = PyDLL(_ctypes_test.__file__)._testfunc_pylist_append
    PyList_Append.restype = c_int
    PyList_Append.argtypes = [py_object, py_object, X]
    trace = []
    trace.append(1)
    PyList_Append(trace, 3, 'dummy')
    trace.append(5)
    self.assertEqual(trace, [1, 2, 3, 4, 5])

    class Y(Structure):
        """This struct size is > sizeof(void*)."""
        _fields_ = [('a', c_void_p), ('b', c_void_p)]

        def __del__(self):
            trace.append(4)

        @classmethod
        def from_param(cls, value):
            trace.append(2)
            return cls()
    PyList_Append = PyDLL(_ctypes_test.__file__)._testfunc_pylist_append
    PyList_Append.restype = c_int
    PyList_Append.argtypes = [py_object, py_object, Y]
    trace = []
    trace.append(1)
    PyList_Append(trace, 3, 'dummy')
    trace.append(5)
    self.assertEqual(trace, [1, 2, 3, 4, 5])