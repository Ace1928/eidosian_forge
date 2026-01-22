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
def fz_invert_pixmap_rect(self, rect):
    """
        Class-aware wrapper for `::fz_invert_pixmap_rect()`.
        	Invert all the pixels in a given rectangle of a (premultiplied)
        	pixmap. All components of all pixels in the rectangle are
        	inverted (except alpha, which is unchanged).
        """
    return _mupdf.FzPixmap_fz_invert_pixmap_rect(self, rect)