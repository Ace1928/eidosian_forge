from __future__ import print_function
import sys
import textwrap
import functools
import itertools
import collections
import gdb
from Cython.Debugger import libpython
def get_remote_cython_globals_dict(self):
    m = gdb.parse_and_eval('__pyx_m')
    try:
        PyModuleObject = gdb.lookup_type('PyModuleObject')
    except RuntimeError:
        raise gdb.GdbError(textwrap.dedent('                Unable to lookup type PyModuleObject, did you compile python\n                with debugging support (-g)?'))
    m = m.cast(PyModuleObject.pointer())
    return m['md_dict']