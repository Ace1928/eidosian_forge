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
def ll_fz_drop_compressed_buffer(buf):
    """
    Low-level wrapper for `::fz_drop_compressed_buffer()`.
    Drop a reference to a compressed buffer. Destroys the buffer
    and frees any storage/other references held by it.

    Never throws exceptions.
    """
    return _mupdf.ll_fz_drop_compressed_buffer(buf)