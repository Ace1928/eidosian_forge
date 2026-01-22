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
def ll_fz_drop_outline(outline):
    """
    Low-level wrapper for `::fz_drop_outline()`.
    Decrements the reference count. When the reference point
    reaches zero, the outline is freed.

    When freed, it will drop linked	outline entries (next and down)
    too, thus a whole outline structure can be dropped by dropping
    the top entry.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_outline(outline)