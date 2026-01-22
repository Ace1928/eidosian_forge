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
def fz_open_document_with_buffer(magic, buffer):
    """
    Class-aware wrapper for `::fz_open_document_with_buffer()`.
    	Open a document using a buffer rather than opening a file on disk.
    """
    return _mupdf.fz_open_document_with_buffer(magic, buffer)