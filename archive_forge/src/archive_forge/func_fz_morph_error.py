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
def fz_morph_error(fromcode, tocode):
    """
    Class-aware wrapper for `::fz_morph_error()`.
    	Called within a catch block this modifies the current
    	exception's code. If it's of type 'fromcode' it is
    	modified to 'tocode'. Typically used for 'downgrading'
    	exception severity.
    """
    return _mupdf.fz_morph_error(fromcode, tocode)