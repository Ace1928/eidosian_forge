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
def fz_calloc_no_throw(count, size):
    """
    Class-aware wrapper for `::fz_calloc_no_throw()`.
    	fz_calloc equivalent that returns NULL rather than throwing
    	exceptions.
    """
    return _mupdf.fz_calloc_no_throw(count, size)