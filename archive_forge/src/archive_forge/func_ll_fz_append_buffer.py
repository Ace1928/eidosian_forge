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
def ll_fz_append_buffer(destination, source):
    """
    Low-level wrapper for `::fz_append_buffer()`.
    Append the contents of the source buffer onto the end of the
    destination buffer, extending automatically as required.

    Ownership of buffers does not change.
    """
    return _mupdf.ll_fz_append_buffer(destination, source)