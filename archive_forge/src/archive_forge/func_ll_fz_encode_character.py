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
def ll_fz_encode_character(font, unicode):
    """
    Low-level wrapper for `::fz_encode_character()`.
    Find the glyph id for a given unicode
    character within a font.

    font: The font to look for the unicode character in.

    unicode: The unicode character to encode.

    Returns the glyph id for the given unicode value, or 0 if
    unknown.
    """
    return _mupdf.ll_fz_encode_character(font, unicode)