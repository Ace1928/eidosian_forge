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
def fz_load_page(self, number):
    """
        Class-aware wrapper for `::fz_load_page()`.
        	Load a given page number from a document. This may be much less
        	efficient than loading by location (chapter+page) for some
        	document types.
        """
    return _mupdf.FzDocument_fz_load_page(self, number)