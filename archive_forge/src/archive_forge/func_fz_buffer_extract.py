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
def fz_buffer_extract(self, data):
    """
        Class-aware wrapper for `::fz_buffer_extract()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_buffer_extract()` => `(size_t, unsigned char *data)`

        	Take ownership of buffer contents.

        	Performs the same task as fz_buffer_storage, but ownership of
        	the data buffer returns with this call. The buffer is left
        	empty.

        	Note: Bad things may happen if this is called on a buffer with
        	multiple references that is being used from multiple threads.

        	data: Pointer to place to retrieve data pointer.

        	Returns length of stream.
        """
    return _mupdf.FzBuffer_fz_buffer_extract(self, data)