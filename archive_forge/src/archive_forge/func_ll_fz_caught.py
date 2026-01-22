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
def ll_fz_caught():
    """
    Low-level wrapper for `::fz_caught()`.
    Within an fz_catch() block, retrieve the error code for
    the current exception.

    This assumes no intervening use of fz_try/fz_catch.
    """
    return _mupdf.ll_fz_caught()