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
def fz_open_document_with_stream(magic, stream):
    """
    Class-aware wrapper for `::fz_open_document_with_stream()`.
    	Open a document using the specified stream object rather than
    	opening a file on disk.

    	magic: a string used to detect document type; either a file name
    	or mime-type.

    	stream: a stream representing the contents of the document file.

    	NOTE: The caller retains ownership of 'stream' - the document will take its
    	own reference if required.
    """
    return _mupdf.fz_open_document_with_stream(magic, stream)