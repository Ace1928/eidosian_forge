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
def ll_fz_advance_glyph(font, glyph, wmode):
    """
    Low-level wrapper for `::fz_advance_glyph()`.
    Return the advance for a given glyph.

    font: The font to look for the glyph in.

    glyph: The glyph to find the advance for.

    wmode: 1 for vertical mode, 0 for horizontal.

    Returns the advance for the glyph.
    """
    return _mupdf.ll_fz_advance_glyph(font, glyph, wmode)