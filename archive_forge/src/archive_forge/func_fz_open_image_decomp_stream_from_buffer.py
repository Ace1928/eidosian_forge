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
def fz_open_image_decomp_stream_from_buffer(self, l2factor):
    """
        Class-aware wrapper for `::fz_open_image_decomp_stream_from_buffer()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_open_image_decomp_stream_from_buffer()` => `(fz_stream *, int l2factor)`

        	Open a stream to read the decompressed version of a buffer,
        	with optional log2 subsampling.

        	l2factor = NULL for no subsampling, or a pointer to an integer
        	containing the maximum log2 subsample factor acceptable (0 =
        	none, 1 = halve dimensions, 2 = quarter dimensions etc). If
        	non-NULL, then *l2factor will be updated on exit with the actual
        	log2 subsample factor achieved.
        """
    return _mupdf.FzCompressedBuffer_fz_open_image_decomp_stream_from_buffer(self, l2factor)