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
def ll_fz_open_document_with_stream_and_dir(magic, stream, dir):
    """
    Low-level wrapper for `::fz_open_document_with_stream_and_dir()`.
    Open a document using the specified stream object rather than
    opening a file on disk.

    magic: a string used to detect document type; either a file name
    or mime-type.

    stream: a stream representing the contents of the document file.

    dir: a 'directory context' for those filetypes that need it.

    NOTE: The caller retains ownership of 'stream' and 'dir' - the document will
    take its own references if required.
    """
    return _mupdf.ll_fz_open_document_with_stream_and_dir(magic, stream, dir)