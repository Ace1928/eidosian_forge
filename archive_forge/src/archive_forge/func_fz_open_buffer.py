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
def fz_open_buffer(self):
    """
        Class-aware wrapper for `::fz_open_buffer()`.
        	Open a buffer as a stream.

        	buf: The buffer to open. Ownership of the buffer is NOT passed
        	in (this function takes its own reference).

        	Returns pointer to newly created stream. May throw exceptions on
        	failure to allocate.
        """
    return _mupdf.FzBuffer_fz_open_buffer(self)