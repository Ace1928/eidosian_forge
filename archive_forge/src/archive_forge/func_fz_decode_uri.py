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
def fz_decode_uri(s):
    """
     Class-aware wrapper for `::fz_decode_uri()`.
    Return a new string representing the unencoded version of the given URI.
    This decodes all escape sequences except those that would result in a reserved
    character that are part of the URI syntax (; / ? : @ & = + $ , #).
    """
    return _mupdf.fz_decode_uri(s)