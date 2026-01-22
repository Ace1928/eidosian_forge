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
def ll_fz_device_gray():
    """
    Low-level wrapper for `::fz_device_gray()`.
    Retrieve global default colorspaces.

    These return borrowed references that should not be dropped,
    unless they are kept first.
    """
    return _mupdf.ll_fz_device_gray()