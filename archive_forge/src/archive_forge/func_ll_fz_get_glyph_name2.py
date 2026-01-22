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
def ll_fz_get_glyph_name2(font, glyph):
    """
     Low-level wrapper for `::fz_get_glyph_name2()`.
    C++ alternative to fz_get_glyph_name() that returns information in a std::string.
    """
    return _mupdf.ll_fz_get_glyph_name2(font, glyph)