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
def fz_skip(self, len):
    """
        Class-aware wrapper for `::fz_skip()`.
        	Read from a stream discarding data.

        	stm: The stream to read from.

        	len: The number of bytes to read.

        	Returns the number of bytes read. May throw exceptions.
        """
    return _mupdf.FzStream_fz_skip(self, len)