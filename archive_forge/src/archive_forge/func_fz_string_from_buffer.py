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
def fz_string_from_buffer(self):
    """
        Class-aware wrapper for `::fz_string_from_buffer()`.
        	Ensure that a buffer's data ends in a
        	0 byte, and return a pointer to it.
        """
    return _mupdf.FzBuffer_fz_string_from_buffer(self)