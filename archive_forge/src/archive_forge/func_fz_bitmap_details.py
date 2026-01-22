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
def fz_bitmap_details(self, w, h, n, stride):
    """
        Class-aware wrapper for `::fz_bitmap_details()`.

        This method has out-params. Python/C# wrappers look like:
        	`fz_bitmap_details()` => `(int w, int h, int n, int stride)`

        	Retrieve details of a given bitmap.

        	bitmap: The bitmap to query.

        	w: Pointer to storage to retrieve width (or NULL).

        	h: Pointer to storage to retrieve height (or NULL).

        	n: Pointer to storage to retrieve number of color components (or
        	NULL).

        	stride: Pointer to storage to retrieve bitmap stride (or NULL).
        """
    return _mupdf.FzBitmap_fz_bitmap_details(self, w, h, n, stride)