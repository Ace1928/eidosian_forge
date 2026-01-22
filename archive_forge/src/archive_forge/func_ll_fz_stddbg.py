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
def ll_fz_stddbg():
    """
    Low-level wrapper for `::fz_stddbg()`.
    Retrieve an fz_output for the default debugging stream. On
    Windows this will be OutputDebugString for non-console apps.
    Otherwise, it is always fz_stderr.

    Optionally may be fz_dropped when finished with.
    """
    return _mupdf.ll_fz_stddbg()