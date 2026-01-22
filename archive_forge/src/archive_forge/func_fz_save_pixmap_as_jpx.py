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
def fz_save_pixmap_as_jpx(self, filename, q):
    """
        Class-aware wrapper for `::fz_save_pixmap_as_jpx()`.
        	Save pixmap data as JP2K with no subsampling.

        	quality = 100 = lossless
        	otherwise for a factor of x compression use 100-x. (so 80 is 1:20 compression)
        """
    return _mupdf.FzPixmap_fz_save_pixmap_as_jpx(self, filename, q)