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
def ll_fz_drop_band_writer(writer):
    """
    Low-level wrapper for `::fz_drop_band_writer()`.
    Drop the reference to the band writer, causing it to be
    destroyed.

    Never throws an exception.
    """
    return _mupdf.ll_fz_drop_band_writer(writer)