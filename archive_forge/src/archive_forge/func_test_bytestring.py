import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version
def test_bytestring(self):
    bytestring = self.alloc_bytestring(b'spam')
    if inferior_python_version < (3, 0):
        bytestring_class = libpython.PyStringObjectPtr
        expected = repr(b'spam')
    else:
        bytestring_class = libpython.PyBytesObjectPtr
        expected = "b'spam'"
    self.assertEqual(type(bytestring), bytestring_class)
    self.assertEqual(self.get_repr(bytestring), expected)