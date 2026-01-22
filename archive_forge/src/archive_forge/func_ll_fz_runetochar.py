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
def ll_fz_runetochar(str, rune):
    """
    Low-level wrapper for `::fz_runetochar()`.
    UTF8 encode a rune to a sequence of chars.

    str: Pointer to a place to put the UTF8 encoded character.

    rune: Pointer to a 'rune'.

    Returns the number of bytes the rune took to output.
    """
    return _mupdf.ll_fz_runetochar(str, rune)