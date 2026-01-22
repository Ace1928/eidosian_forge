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
def ll_fz_runelen(rune):
    """
    Low-level wrapper for `::fz_runelen()`.
    Count how many chars are required to represent a rune.

    rune: The rune to encode.

    Returns the number of bytes required to represent this run in
    UTF8.
    """
    return _mupdf.ll_fz_runelen(rune)