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
def ll_fz_last_page(doc):
    """
    Low-level wrapper for `::fz_last_page()`.
    Function to get the location for the last page in the document.
    Using this can be far more efficient in some cases than calling
    fz_count_pages and using the page number.
    """
    return _mupdf.ll_fz_last_page(doc)