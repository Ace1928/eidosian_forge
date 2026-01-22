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
def ll_fz_new_stext_page_from_page(page, options):
    """
    Low-level wrapper for `::fz_new_stext_page_from_page()`.
    Extract text from page.

    Ownership of the fz_stext_page is returned to the caller.
    """
    return _mupdf.ll_fz_new_stext_page_from_page(page, options)