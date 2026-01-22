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
def ll_fz_write_pixmap_as_pdfocr(out, pixmap, options):
    """
    Low-level wrapper for `::fz_write_pixmap_as_pdfocr()`.
    Write a (Greyscale or RGB) pixmap as pdfocr.
    """
    return _mupdf.ll_fz_write_pixmap_as_pdfocr(out, pixmap, options)