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
def ll_fz_font_shaper_data(font):
    """
    Low-level wrapper for `::fz_font_shaper_data()`.
    Retrieve a pointer to the shaper data
    structure for the given font.

    font: The font to query.

    Returns a pointer to the shaper data structure (or NULL if
    font is NULL).
    """
    return _mupdf.ll_fz_font_shaper_data(font)