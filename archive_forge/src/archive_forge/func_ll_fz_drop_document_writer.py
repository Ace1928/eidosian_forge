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
def ll_fz_drop_document_writer(wri):
    """
    Low-level wrapper for `::fz_drop_document_writer()`.
    Called to discard a fz_document_writer.
    This may be called at any time during the process to release all
    the resources owned by the writer.

    Calling drop without having previously called close may leave
    the file in an inconsistent state.
    """
    return _mupdf.ll_fz_drop_document_writer(wri)