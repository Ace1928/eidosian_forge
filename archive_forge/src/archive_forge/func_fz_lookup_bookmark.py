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
def fz_lookup_bookmark(self, mark):
    """
        Class-aware wrapper for `::fz_lookup_bookmark()`.
        	Find a bookmark and return its page number.
        """
    return _mupdf.FzDocument_fz_lookup_bookmark(self, mark)