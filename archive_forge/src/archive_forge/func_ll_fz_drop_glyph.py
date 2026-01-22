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
def ll_fz_drop_glyph(pix):
    """
    Low-level wrapper for `::fz_drop_glyph()`.
    Drop a reference and free a glyph.

    Decrement the reference count for the glyph. When no
    references remain the glyph will be freed.
    """
    return _mupdf.ll_fz_drop_glyph(pix)