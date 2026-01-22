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
def fz_new_pixmap_with_bbox(self, bbox, seps, alpha):
    """
        Class-aware wrapper for `::fz_new_pixmap_with_bbox()`.
        	Create a pixmap of a given size, location and pixel format.

        	The bounding box specifies the size of the created pixmap and
        	where it will be located. The colorspace determines the number
        	of components per pixel. Alpha is always present. Pixmaps are
        	reference counted, so drop references using fz_drop_pixmap.

        	colorspace: Colorspace format used for the created pixmap. The
        	pixmap will keep a reference to the colorspace.

        	bbox: Bounding box specifying location/size of created pixmap.

        	seps: Details of separations.

        	alpha: 0 for no alpha, 1 for alpha.

        	Returns a pointer to the new pixmap. Throws exception on failure
        	to allocate.
        """
    return _mupdf.FzColorspace_fz_new_pixmap_with_bbox(self, bbox, seps, alpha)