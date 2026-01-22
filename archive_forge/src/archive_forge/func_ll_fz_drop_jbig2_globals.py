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
def ll_fz_drop_jbig2_globals(globals):
    """
    Low-level wrapper for `::fz_drop_jbig2_globals()`.
    Decrement the reference count for a jbig2 globals record.
    When the reference count hits zero, the record is freed.

    Never throws an exception.
    """
    return _mupdf.ll_fz_drop_jbig2_globals(globals)