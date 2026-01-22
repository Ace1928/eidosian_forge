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
def fz_compress_ccitt_fax_g4(data, columns, rows, stride):
    """
    Class-aware wrapper for `::fz_compress_ccitt_fax_g4()`.
    	Compress bitmap data as CCITT Group 4 2D fax image.
    	Creates a stream assuming the default PDF parameters, except
    	K=-1 and the number of columns.
    """
    return _mupdf.fz_compress_ccitt_fax_g4(data, columns, rows, stride)