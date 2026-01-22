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
def ll_fz_recognize_document_content(filename):
    """
    Low-level wrapper for `::fz_recognize_document_content()`.
    Given a filename find a document handler that can handle a
    document of this type.

    filename: The filename of the document. This will be opened and sampled
    to check data.
    """
    return _mupdf.ll_fz_recognize_document_content(filename)