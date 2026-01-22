from __future__ import print_function
import gdb
import os
import locale
import sys
import sys
import libpython
import re
import warnings
import tempfile
import functools
import textwrap
import itertools
import traceback
def evalcode(self, code, input_type, global_dict=None, local_dict=None):
    """
        Evaluate python code `code` given as a string in the inferior and
        return the result as a gdb.Value. Returns a new reference in the
        inferior.

        Of course, executing any code in the inferior may be dangerous and may
        leave the debuggee in an unsafe state or terminate it altogether.
        """
    if '\x00' in code:
        raise gdb.GdbError('String contains NUL byte.')
    code += '\x00'
    pointer = self.alloc_string(code)
    globalsp = pointervalue(global_dict)
    localsp = pointervalue(local_dict)
    if globalsp == 0 or localsp == 0:
        raise gdb.GdbError('Unable to obtain or create locals or globals.')
    code = '\n            PyRun_String(\n                (char *) %(code)d,\n                (int) %(start)d,\n                (PyObject *) %(globals)s,\n                (PyObject *) %(locals)d)\n        ' % dict(code=pointer, start=input_type, globals=globalsp, locals=localsp)
    with FetchAndRestoreError():
        try:
            pyobject_return_value = gdb.parse_and_eval(code)
        finally:
            self.free(pointer)
    return pyobject_return_value