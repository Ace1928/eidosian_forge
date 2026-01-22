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
def fz_write_pixmap_as_jpeg(self, pix, quality, invert_cmyk):
    """
        Class-aware wrapper for `::fz_write_pixmap_as_jpeg()`.
        	Write a pixmap as a JPEG.
        """
    return _mupdf.FzOutput_fz_write_pixmap_as_jpeg(self, pix, quality, invert_cmyk)