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
def ll_fz_new_buffer_from_pixmap_as_png(pixmap, color_params):
    """
    Low-level wrapper for `::fz_new_buffer_from_pixmap_as_png()`.
    Reencode a given pixmap as a PNG into a buffer.

    Ownership of the buffer is returned.
    """
    return _mupdf.ll_fz_new_buffer_from_pixmap_as_png(pixmap, color_params)