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
def ll_fz_outline_glyph(font, gid, ctm):
    """
    Low-level wrapper for `::fz_outline_glyph()`.
    Look a glyph up from a font, and return the outline of the
    glyph using the given transform.

    The caller owns the returned path, and so is responsible for
    ensuring that it eventually gets dropped.
    """
    return _mupdf.ll_fz_outline_glyph(font, gid, ctm)