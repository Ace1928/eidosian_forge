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
def fz_pixmap_components(self):
    """
        Class-aware wrapper for `::fz_pixmap_components()`.
        	Return the number of components in a pixmap.

        	Returns the number of components (including spots and alpha).
        """
    return _mupdf.FzPixmap_fz_pixmap_components(self)