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
def fz_string_from_text_language(str, lang):
    """
    Class-aware wrapper for `::fz_string_from_text_language()`.
    	Recover ISO 639 (639-{1,2,3,5}) language specification
    	strings losslessly from a 15 bit fz_text_language code.

    	No validation is carried out. See note above.
    """
    return _mupdf.fz_string_from_text_language(str, lang)