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
def ll_fz_open_null_filter(chain, len, offset):
    """
    Low-level wrapper for `::fz_open_null_filter()`.
    The null filter reads a specified amount of data from the
    substream.
    """
    return _mupdf.ll_fz_open_null_filter(chain, len, offset)