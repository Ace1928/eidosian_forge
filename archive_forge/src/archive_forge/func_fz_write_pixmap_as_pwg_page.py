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
def fz_write_pixmap_as_pwg_page(self, pixmap, pwg):
    """
        Class-aware wrapper for `::fz_write_pixmap_as_pwg_page()`.
        	Write a pixmap as a PWG page.

        	Caller should provide a file header by calling
        	fz_write_pwg_file_header, but can then write several pages to
        	the same file.
        """
    return _mupdf.FzOutput_fz_write_pixmap_as_pwg_page(self, pixmap, pwg)