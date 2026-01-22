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
def ll_fz_drop_font(font):
    """
    Low-level wrapper for `::fz_drop_font()`.
    Drop a reference to a fz_font, destroying the
    font when the last reference is dropped.

    font: The font to drop a reference to.
    """
    return _mupdf.ll_fz_drop_font(font)