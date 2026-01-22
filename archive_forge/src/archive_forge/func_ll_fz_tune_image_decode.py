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
def ll_fz_tune_image_decode(image_decode, arg):
    """
    Low-level wrapper for `::fz_tune_image_decode()`.
    Set the tuning function to use for
    image decode.

    image_decode: Function to use.

    arg: Opaque argument to be passed to tuning function.
    """
    return _mupdf.ll_fz_tune_image_decode(image_decode, arg)