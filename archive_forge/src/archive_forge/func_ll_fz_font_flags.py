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
def ll_fz_font_flags(font):
    """
    Low-level wrapper for `::fz_font_flags()`.
    Retrieve a pointer to the font flags
    for a given font. These can then be updated as required.

    font: The font to query

    Returns a pointer to the flags structure (or NULL, if
    the font is NULL).
    """
    return _mupdf.ll_fz_font_flags(font)