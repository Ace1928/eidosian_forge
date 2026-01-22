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
def ll_fz_pre_shear(m, sx, sy):
    """
    Low-level wrapper for `::fz_pre_shear()`.
    Premultiply a matrix with a shearing matrix.

    The shearing matrix is of the form [ 1 sy sx 1 0 0 ].

    m: pointer to matrix to premultiply

    sx, sy: Shearing factors. A shearing factor of 0.0 will not
    cause any shearing along the relevant axis.

    Returns m (updated).
    """
    return _mupdf.ll_fz_pre_shear(m, sx, sy)