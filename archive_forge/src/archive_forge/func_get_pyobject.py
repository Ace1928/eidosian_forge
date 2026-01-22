import gdb
from Cython.Debugger import libcython
from Cython.Debugger import libpython
from . import test_libcython_in_gdb
from .test_libcython_in_gdb import inferior_python_version
def get_pyobject(self, code):
    value = gdb.parse_and_eval(code)
    assert libpython.pointervalue(value) != 0
    return value