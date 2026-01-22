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
@classmethod
def from_pyobject_ptr(cls, gdbval):
    """
        Try to locate the appropriate derived class dynamically, and cast
        the pointer accordingly.
        """
    try:
        p = PyObjectPtr(gdbval)
        cls = cls.subclass_from_type(p.type())
        return cls(gdbval, cast_to=cls.get_gdb_type())
    except RuntimeError:
        pass
    return cls(gdbval)