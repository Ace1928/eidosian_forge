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
def fz_deflate(dest, compressed_length, source, source_length, level):
    """
     Class-aware wrapper for `::fz_deflate()`.

    	This function has out-params. Python/C# wrappers look like:
    		`fz_deflate(unsigned char *dest, const unsigned char *source, size_t source_length, ::fz_deflate_level level)` => size_t compressed_length

    		Compress source_length bytes of data starting
    		at source, into a buffer of length *destLen, starting at dest.
    compressed_length will be updated on exit to contain the size
    		actually used.
    """
    return _mupdf.fz_deflate(dest, compressed_length, source, source_length, level)