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
def ll_fz_copy_option(val, dest, maxlen):
    """
    Low-level wrapper for `::fz_copy_option()`.
    Copy an option (val) into a destination buffer (dest), of maxlen
    bytes.

    Returns the number of bytes (including terminator) that did not
    fit. If val is maxlen or greater bytes in size, it will be left
    unterminated.
    """
    return _mupdf.ll_fz_copy_option(val, dest, maxlen)