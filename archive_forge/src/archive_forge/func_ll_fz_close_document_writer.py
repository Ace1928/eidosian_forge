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
def ll_fz_close_document_writer(wri):
    """
    Low-level wrapper for `::fz_close_document_writer()`.
    Called to end the process of writing
    pages to a document.

    This writes any file level trailers required. After this
    completes successfully the file is up to date and complete.
    """
    return _mupdf.ll_fz_close_document_writer(wri)