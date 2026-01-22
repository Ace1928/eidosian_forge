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
def ll_fz_pixmap_spots(pix):
    """
    Low-level wrapper for `::fz_pixmap_spots()`.
    Return the number of spots in a pixmap.

    Returns the number of spots (components, less colorants and
    alpha). Does not throw exceptions.
    """
    return _mupdf.ll_fz_pixmap_spots(pix)