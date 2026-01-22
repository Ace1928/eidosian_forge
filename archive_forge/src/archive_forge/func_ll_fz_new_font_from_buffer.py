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
def ll_fz_new_font_from_buffer(name, buffer, index, use_glyph_bbox):
    """
    Low-level wrapper for `::fz_new_font_from_buffer()`.
    Create a new font from a font file in a fz_buffer.

    Fonts created in this way, will be eligible for embedding by default.

    name: Name of font (leave NULL to use name from font).

    buffer: Buffer to load from.

    index: Which font from the file to load (0 for default).

    use_glyph_box: 1 if we should use the glyph bbox, 0 otherwise.

    Returns new font handle, or throws exception on error.
    """
    return _mupdf.ll_fz_new_font_from_buffer(name, buffer, index, use_glyph_bbox)