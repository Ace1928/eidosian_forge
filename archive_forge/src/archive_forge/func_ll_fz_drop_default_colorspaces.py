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
def ll_fz_drop_default_colorspaces(default_cs):
    """
    Low-level wrapper for `::fz_drop_default_colorspaces()`.
    Drop a reference to the default colorspaces structure. When the
    reference count reaches 0, the references it holds internally
    to the underlying colorspaces will be dropped, and the structure
    will be destroyed.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_default_colorspaces(default_cs)