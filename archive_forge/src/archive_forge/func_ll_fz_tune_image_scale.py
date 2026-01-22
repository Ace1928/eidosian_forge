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
def ll_fz_tune_image_scale(image_scale, arg):
    """
    Low-level wrapper for `::fz_tune_image_scale()`.
    Set the tuning function to use for
    image scaling.

    image_scale: Function to use.

    arg: Opaque argument to be passed to tuning function.
    """
    return _mupdf.ll_fz_tune_image_scale(image_scale, arg)