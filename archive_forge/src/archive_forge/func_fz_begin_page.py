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
def fz_begin_page(self, mediabox):
    """
        Class-aware wrapper for `::fz_begin_page()`.
        	Called to start the process of writing a page to
        	a document.

        	mediabox: page size rectangle in points.

        	Returns a borrowed fz_device to write page contents to. This
        	should be kept if required, and only dropped if it was kept.
        """
    return _mupdf.FzDocumentWriter_fz_begin_page(self, mediabox)