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
def ll_fz_write_base64_buffer(out, data, newline):
    """
    Low-level wrapper for `::fz_write_base64_buffer()`.
    Write a base64 encoded fz_buffer, optionally with periodic
    newlines.
    """
    return _mupdf.ll_fz_write_base64_buffer(out, data, newline)