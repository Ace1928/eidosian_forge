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
def fz_hb_unlock():
    """
    Class-aware wrapper for `::fz_hb_unlock()`.
    	Unlock after a Harfbuzz call. This reuses
    	FZ_LOCK_FREETYPE.
    """
    return _mupdf.fz_hb_unlock()