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
def fz_compare_separations(self, sep2):
    """
        Class-aware wrapper for `::fz_compare_separations()`.
        	Compare 2 separations structures (or NULLs).

        	Return 0 if identical, non-zero if not identical.
        """
    return _mupdf.FzSeparations_fz_compare_separations(self, sep2)