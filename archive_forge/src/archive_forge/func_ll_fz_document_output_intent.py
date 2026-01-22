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
def ll_fz_document_output_intent(doc):
    """
    Low-level wrapper for `::fz_document_output_intent()`.
    Find the output intent colorspace if the document has defined
    one.

    Returns a borrowed reference that should not be dropped, unless
    it is kept first.
    """
    return _mupdf.ll_fz_document_output_intent(doc)