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
def fz_new_type3_font(name, matrix):
    """
    Class-aware wrapper for `::fz_new_type3_font()`.
    	Create a new (empty) type3 font.

    	name: Name of font (or NULL).

    	matrix: Font matrix.

    	Returns a new font handle, or throws exception on
    	allocation failure.
    """
    return _mupdf.fz_new_type3_font(name, matrix)