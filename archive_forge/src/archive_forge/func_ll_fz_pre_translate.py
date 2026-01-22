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
def ll_fz_pre_translate(m, tx, ty):
    """
    Low-level wrapper for `::fz_pre_translate()`.
    Translate a matrix by premultiplication.

    m: The matrix to translate

    tx, ty: Translation distances along the X- and Y-axes. A
    translation of 0 will not cause any translation along the
    relevant axis.

    Returns m.
    """
    return _mupdf.ll_fz_pre_translate(m, tx, ty)