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
def ll_fz_font_bbox(font):
    """
    Low-level wrapper for `::fz_font_bbox()`.
    Retrieve the font bbox.

    font: The font to query.

    Returns the font bbox by value; it is valid only if
    fz_font_flags(font)->invalid_bbox is zero.
    """
    return _mupdf.ll_fz_font_bbox(font)