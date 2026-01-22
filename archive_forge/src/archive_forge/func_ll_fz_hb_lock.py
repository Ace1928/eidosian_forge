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
def ll_fz_hb_lock():
    """
    Low-level wrapper for `::fz_hb_lock()`.
    Lock against Harfbuzz being called
    simultaneously in several threads. This reuses
    FZ_LOCK_FREETYPE.
    """
    return _mupdf.ll_fz_hb_lock()