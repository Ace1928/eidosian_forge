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
def fz_trim_buffer(self):
    """
        Class-aware wrapper for `::fz_trim_buffer()`.
        	Trim wasted capacity from a buffer by resizing internal memory.
        """
    return _mupdf.FzBuffer_fz_trim_buffer(self)