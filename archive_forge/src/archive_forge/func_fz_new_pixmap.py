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
def fz_new_pixmap(self, w, h, seps, alpha):
    """
        Class-aware wrapper for `::fz_new_pixmap()`.
        	Create a new pixmap, with its origin at (0,0)

        	cs: The colorspace to use for the pixmap, or NULL for an alpha
        	plane/mask.

        	w: The width of the pixmap (in pixels)

        	h: The height of the pixmap (in pixels)

        	seps: Details of separations.

        	alpha: 0 for no alpha, 1 for alpha.

        	Returns a pointer to the new pixmap. Throws exception on failure
        	to allocate.
        """
    return _mupdf.FzColorspace_fz_new_pixmap(self, w, h, seps, alpha)