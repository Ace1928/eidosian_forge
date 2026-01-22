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
def ll_fz_filter_store(fn, arg, type):
    """
    Low-level wrapper for `::fz_filter_store()`.
    Filter every element in the store with a matching type with the
    given function.

    If the function returns 1 for an element, drop the element.
    """
    return _mupdf.ll_fz_filter_store(fn, arg, type)