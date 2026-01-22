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
def ll_fz_next_page(doc, loc):
    """
    Low-level wrapper for `::fz_next_page()`.
    Function to get the location of the next page (allowing for the
    end of chapters etc). If at the end of the document, returns the
    current location.
    """
    return _mupdf.ll_fz_next_page(doc, loc)