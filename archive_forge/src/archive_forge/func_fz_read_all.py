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
def fz_read_all(self, initial):
    """
        Class-aware wrapper for `::fz_read_all()`.
        	Read all of a stream into a buffer.

        	stm: The stream to read from

        	initial: Suggested initial size for the buffer.

        	Returns a buffer created from reading from the stream. May throw
        	exceptions on failure to allocate.
        """
    return _mupdf.FzStream_fz_read_all(self, initial)