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
def ll_fz_open_rld(chain):
    """
    Low-level wrapper for `::fz_open_rld()`.
    rld filter performs Run Length Decoding of data read
    from the chained filter.
    """
    return _mupdf.ll_fz_open_rld(chain)