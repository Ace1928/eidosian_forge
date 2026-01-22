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
def ll_fz_read_utf16_le(stm):
    """
    Low-level wrapper for `::fz_read_utf16_le()`.
    Read a utf-16 rune from a stream. (little endian and
    big endian respectively).

    In the event of encountering badly formatted utf-16 codes
    (mismatched surrogates) no error/exception is given, but
    undefined values may be returned.
    """
    return _mupdf.ll_fz_read_utf16_le(stm)