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
def ll_fz_is_eof(stm):
    """
    Low-level wrapper for `::fz_is_eof()`.
    Query if the stream has reached EOF (during normal bytewise
    reading).

    See fz_is_eof_bits for the equivalent function for bitwise
    reading.
    """
    return _mupdf.ll_fz_is_eof(stm)