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
def fz_option_eq(a, b):
    """
    Class-aware wrapper for `::fz_option_eq()`.
    	Check to see if an option, a, from a string matches a reference
    	option, b.

    	(i.e. a could be 'foo' or 'foo,bar...' etc, but b can only be
    	'foo'.)
    """
    return _mupdf.fz_option_eq(a, b)