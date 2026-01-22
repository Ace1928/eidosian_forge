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
def ll_fz_try_open_file(name):
    """
    Low-level wrapper for `::fz_try_open_file()`.
    Open the named file and wrap it in a stream.

    Does the same as fz_open_file, but in the event the file
    does not open, it will return NULL rather than throw an
    exception.
    """
    return _mupdf.ll_fz_try_open_file(name)