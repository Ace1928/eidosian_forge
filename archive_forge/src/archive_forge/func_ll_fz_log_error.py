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
def ll_fz_log_error(str):
    """
    Low-level wrapper for `::fz_log_error()`.
    Log a (preformatted) string to the registered
    error stream (stderr by default).
    """
    return _mupdf.ll_fz_log_error(str)