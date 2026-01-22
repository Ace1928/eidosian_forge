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
def ll_fz_drop_buffer(buf):
    """
    Low-level wrapper for `::fz_drop_buffer()`.
    Drop a reference to the buffer. When the reference count reaches
    zero, the buffer is destroyed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_buffer(buf)