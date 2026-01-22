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
def ll_fz_post_scale(m, sx, sy):
    """
    Low-level wrapper for `::fz_post_scale()`.
    Scale a matrix by postmultiplication.

    m: Pointer to the matrix to scale

    sx, sy: Scaling factors along the X- and Y-axes. A scaling
    factor of 1.0 will not cause any scaling along the relevant
    axis.

    Returns m (updated).
    """
    return _mupdf.ll_fz_post_scale(m, sx, sy)