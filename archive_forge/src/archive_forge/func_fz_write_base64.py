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
def fz_write_base64(self, data, size, newline):
    """
        Class-aware wrapper for `::fz_write_base64()`.
        	Write a base64 encoded data block, optionally with periodic
        	newlines.
        """
    return _mupdf.FzOutput_fz_write_base64(self, data, size, newline)