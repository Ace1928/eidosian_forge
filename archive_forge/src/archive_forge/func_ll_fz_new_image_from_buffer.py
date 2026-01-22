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
def ll_fz_new_image_from_buffer(buffer):
    """
    Low-level wrapper for `::fz_new_image_from_buffer()`.
    Create a new image from a
    buffer of data, inferring its type from the format
    of the data.
    """
    return _mupdf.ll_fz_new_image_from_buffer(buffer)