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
def fz_end_page(self):
    """
        Class-aware wrapper for `::fz_end_page()`.
        	Called to end the process of writing a page to a
        	document.
        """
    return _mupdf.FzDocumentWriter_fz_end_page(self)