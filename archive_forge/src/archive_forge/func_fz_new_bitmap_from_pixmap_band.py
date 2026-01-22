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
def fz_new_bitmap_from_pixmap_band(self, ht, band_start):
    """
        Class-aware wrapper for `::fz_new_bitmap_from_pixmap_band()`.
        	Make a bitmap from a pixmap and a
        	halftone, allowing for the position of the pixmap within an
        	overall banded rendering.

        	pix: The pixmap to generate from. Currently must be a single
        	color component with no alpha.

        	ht: The halftone to use. NULL implies the default halftone.

        	band_start: Vertical offset within the overall banded rendering
        	(in pixels)

        	Returns the resultant bitmap. Throws exceptions in the case of
        	failure to allocate.
        """
    return _mupdf.FzPixmap_fz_new_bitmap_from_pixmap_band(self, ht, band_start)