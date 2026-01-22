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
def pdf_load_encoding(estrings, encoding):
    """
    Class-aware wrapper for `::pdf_load_encoding()`.

    This function has out-params. Python/C# wrappers look like:
    	`pdf_load_encoding(const char *encoding)` => const char *estrings
    """
    return _mupdf.pdf_load_encoding(estrings, encoding)