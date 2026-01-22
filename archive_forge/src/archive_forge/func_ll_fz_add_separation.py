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
def ll_fz_add_separation(sep, name, cs, cs_channel):
    """
    Low-level wrapper for `::fz_add_separation()`.
    Add a separation (null terminated name, colorspace)
    """
    return _mupdf.ll_fz_add_separation(sep, name, cs, cs_channel)