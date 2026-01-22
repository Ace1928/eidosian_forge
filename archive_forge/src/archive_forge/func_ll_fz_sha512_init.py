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
def ll_fz_sha512_init(state):
    """
    Low-level wrapper for `::fz_sha512_init()`.
    SHA512 initialization. Begins an SHA512 operation, initialising
    the supplied context.

    Never throws an exception.
    """
    return _mupdf.ll_fz_sha512_init(state)