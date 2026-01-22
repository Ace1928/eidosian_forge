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
def fz_memrnd(block, len):
    """
    Class-aware wrapper for `::fz_memrnd()`.
    	Fill block with len bytes of pseudo-randomness.
    """
    return _mupdf.fz_memrnd(block, len)