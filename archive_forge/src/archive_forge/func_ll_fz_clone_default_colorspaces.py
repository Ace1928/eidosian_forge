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
def ll_fz_clone_default_colorspaces(base):
    """
    Low-level wrapper for `::fz_clone_default_colorspaces()`.
    Returns a reference to a newly cloned default colorspaces
    structure.

    The new clone may safely be altered without fear of race
    conditions as the caller is the only reference holder.
    """
    return _mupdf.ll_fz_clone_default_colorspaces(base)