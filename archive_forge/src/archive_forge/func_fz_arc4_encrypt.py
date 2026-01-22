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
def fz_arc4_encrypt(self, dest, src, len):
    """
        Class-aware wrapper for `::fz_arc4_encrypt()`.
        	RC4 block encrypt operation; encrypt src into dst (both of
        	length len) updating the RC4 state as we go.

        	Never throws an exception.
        """
    return _mupdf.FzArc4_fz_arc4_encrypt(self, dest, src, len)