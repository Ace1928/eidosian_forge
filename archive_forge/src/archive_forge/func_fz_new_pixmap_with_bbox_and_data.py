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
def fz_new_pixmap_with_bbox_and_data(self, rect, seps, alpha, samples):
    """
        Class-aware wrapper for `::fz_new_pixmap_with_bbox_and_data()`.
        	Create a pixmap of a given size, location and pixel format,
        	using the supplied data block.

        	The bounding box specifies the size of the created pixmap and
        	where it will be located. The colorspace determines the number
        	of components per pixel. Alpha is always present. Pixmaps are
        	reference counted, so drop references using fz_drop_pixmap.

        	colorspace: Colorspace format used for the created pixmap. The
        	pixmap will keep a reference to the colorspace.

        	rect: Bounding box specifying location/size of created pixmap.

        	seps: Details of separations.

        	alpha: Number of alpha planes (0 or 1).

        	samples: The data block to keep the samples in.

        	Returns a pointer to the new pixmap. Throws exception on failure
        	to allocate.
        """
    return _mupdf.FzColorspace_fz_new_pixmap_with_bbox_and_data(self, rect, seps, alpha, samples)