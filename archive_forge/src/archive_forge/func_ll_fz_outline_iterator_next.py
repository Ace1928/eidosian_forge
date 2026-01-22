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
def ll_fz_outline_iterator_next(iter):
    """
    Low-level wrapper for `::fz_outline_iterator_next()`.
    Calls to move the iterator position.

    A negative return value means we could not move as requested. Otherwise:
    0 = the final position has a valid item.
    1 = not a valid item, but we can insert an item here.
    """
    return _mupdf.ll_fz_outline_iterator_next(iter)