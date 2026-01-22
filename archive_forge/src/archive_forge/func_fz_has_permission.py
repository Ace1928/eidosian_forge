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
def fz_has_permission(self, p):
    """
        Class-aware wrapper for `::fz_has_permission()`.
        	Check permission flags on document.
        """
    return _mupdf.FzDocument_fz_has_permission(self, p)