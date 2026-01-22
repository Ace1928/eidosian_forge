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
def fz_runeidx(str, p):
    """
    Class-aware wrapper for `::fz_runeidx()`.
    	Compute the index of a rune in a string.

    	str: Pointer to beginning of a string.

    	p: Pointer to a char in str.

    	Returns the index of the rune pointed to by p in str.
    """
    return _mupdf.fz_runeidx(str, p)