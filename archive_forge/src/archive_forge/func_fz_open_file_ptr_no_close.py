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
def fz_open_file_ptr_no_close(file):
    """
    Class-aware wrapper for `::fz_open_file_ptr_no_close()`.
    	Create a stream from a FILE * that will not be closed
    	when the stream is dropped.
    """
    return _mupdf.fz_open_file_ptr_no_close(file)