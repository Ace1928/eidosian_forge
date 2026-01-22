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
def ll_fz_is_point_inside_irect(x, y, r):
    """
    Low-level wrapper for `::fz_is_point_inside_irect()`.
    Inclusion test for irects. (Rect is assumed to be open, i.e.
    top right corner is not included).
    """
    return _mupdf.ll_fz_is_point_inside_irect(x, y, r)