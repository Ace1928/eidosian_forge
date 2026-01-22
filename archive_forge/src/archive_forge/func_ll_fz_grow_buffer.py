from sys import version_info as _swig_python_version_info
import weakref
import inspect
import os
import re
import sys
import traceback
import inspect
import io
import os
import sys
import traceback
import types
def ll_fz_grow_buffer(buf):
    """
    Low-level wrapper for `::fz_grow_buffer()`.
    Make some space within a buffer (i.e. ensure that
    capacity > size).
    """
    return _mupdf.ll_fz_grow_buffer(buf)