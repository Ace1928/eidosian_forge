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
def ll_fz_samples_get(pixmap, offset):
    """
     Low-level wrapper for `::fz_samples_get()`.
    Provides simple (but slow) access to pixmap data from Python and C#.
    """
    return _mupdf.ll_fz_samples_get(pixmap, offset)