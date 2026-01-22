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
def fz_new_bitmap_from_pixmap(self, ht):
    """
        Class-aware wrapper for `::fz_new_bitmap_from_pixmap()`.
        	Make a bitmap from a pixmap and a halftone.

        	pix: The pixmap to generate from. Currently must be a single
        	color component with no alpha.

        	ht: The halftone to use. NULL implies the default halftone.

        	Returns the resultant bitmap. Throws exceptions in the case of
        	failure to allocate.
        """
    return _mupdf.FzPixmap_fz_new_bitmap_from_pixmap(self, ht)