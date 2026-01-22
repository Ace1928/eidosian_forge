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
def get_pyop(self):
    try:
        f = self._gdbframe.read_var('f')
        frame = PyFrameObjectPtr.from_pyobject_ptr(f)
        if not frame.is_optimized_out():
            return frame
        orig_frame = frame
        caller = self._gdbframe.older()
        if caller:
            f = caller.read_var('f')
            frame = PyFrameObjectPtr.from_pyobject_ptr(f)
            if not frame.is_optimized_out():
                return frame
        return orig_frame
    except ValueError:
        return None