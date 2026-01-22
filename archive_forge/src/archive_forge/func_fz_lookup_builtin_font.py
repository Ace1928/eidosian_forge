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
def fz_lookup_builtin_font(name, bold, italic, len):
    """
    Class-aware wrapper for `::fz_lookup_builtin_font()`.

    This function has out-params. Python/C# wrappers look like:
    	`fz_lookup_builtin_font(const char *name, int bold, int italic)` => `(const unsigned char *, int len)`

    	Search the builtin fonts for a match.
    	Whether a given font is present or not will depend on the
    	configuration in which MuPDF is built.

    	name: The name of the font desired.

    	bold: 1 if bold desired, 0 otherwise.

    	italic: 1 if italic desired, 0 otherwise.

    	len: Pointer to a place to receive the length of the discovered
    	font buffer.

    	Returns a pointer to the font file data, or NULL if not present.
    """
    return _mupdf.fz_lookup_builtin_font(name, bold, italic, len)