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
def ll_fz_clear_buffer(buf):
    """
    Low-level wrapper for `::fz_clear_buffer()`.
    Empties the buffer. Storage is not freed, but is held ready
    to be reused as the buffer is refilled.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_clear_buffer(buf)