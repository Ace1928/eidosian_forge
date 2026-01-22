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
def fz_new_buffer_from_page_with_format(self, format, options, transform, cookie):
    """
        Class-aware wrapper for `::fz_new_buffer_from_page_with_format()`.
        	Returns an fz_buffer containing a page after conversion to specified format.

        	page: The page to convert.
        	format, options: Passed to fz_new_document_writer_with_output() internally.
        	transform, cookie: Passed to fz_run_page() internally.
        """
    return _mupdf.FzPage_fz_new_buffer_from_page_with_format(self, format, options, transform, cookie)