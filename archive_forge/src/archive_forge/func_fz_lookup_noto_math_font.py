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
def fz_lookup_noto_math_font(len):
    """
    Class-aware wrapper for `::fz_lookup_noto_math_font()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_lookup_noto_math_font()` => `(const unsigned char *, int len)`

    	Search the builtin noto fonts specific symbol fonts.
    	Whether a font is present or not will depend on the
    	configuration in which MuPDF is built.
    """
    return _mupdf.fz_lookup_noto_math_font(len)