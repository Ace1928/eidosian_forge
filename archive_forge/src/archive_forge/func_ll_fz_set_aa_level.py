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
def ll_fz_set_aa_level(bits):
    """
    Low-level wrapper for `::fz_set_aa_level()`.
    Set the number of bits of antialiasing we should
    use (for both text and graphics).

    bits: The number of bits of antialiasing to use (values are
    clamped to within the 0 to 8 range).
    """
    return _mupdf.ll_fz_set_aa_level(bits)