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
def ll_fz_new_stroke_state():
    """
    Low-level wrapper for `::fz_new_stroke_state()`.
    Create a new (empty) stroke state structure (with no dash
    data) and return a reference to it.

    Throws exception on failure to allocate.
    """
    return _mupdf.ll_fz_new_stroke_state()