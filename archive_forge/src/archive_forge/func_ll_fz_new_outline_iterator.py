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
def ll_fz_new_outline_iterator(doc):
    """
    Low-level wrapper for `::fz_new_outline_iterator()`.
    Get an iterator for the document outline.

    Should be freed by fz_drop_outline_iterator.
    """
    return _mupdf.ll_fz_new_outline_iterator(doc)