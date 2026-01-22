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
def fz_format_output_path(path, size, fmt, page):
    """
    Class-aware wrapper for `::fz_format_output_path()`.
    	create output file name using a template.

    	If the path contains %[0-9]*d, the first such pattern will be
    	replaced with the page number. If the template does not contain
    	such a pattern, the page number will be inserted before the
    	filename extension. If the template does not have a filename
    	extension, the page number will be added to the end.
    """
    return _mupdf.fz_format_output_path(path, size, fmt, page)