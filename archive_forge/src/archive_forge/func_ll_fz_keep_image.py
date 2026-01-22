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
def ll_fz_keep_image(image):
    """
    Low-level wrapper for `::fz_keep_image()`.
    Increment the (normal) reference count for an image. Returns the
    same pointer.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_keep_image(image)