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
def ll_fz_convert_color(ss, sv, ds, is_, params):
    """
    Low-level Python version of fz_convert_color().

    `sv` should be a float or list of 1-4 floats or a SWIG
    representation of a float*.

    Returns (dv0, dv1, dv2, dv3).
    """
    dv = fz_convert_color2_v()
    if isinstance(sv, float):
        ll_fz_convert_color2(ss, sv, 0.0, 0.0, 0.0, ds, dv, is_, params)
    elif isinstance(sv, (tuple, list)):
        sv2 = tuple(sv) + (0,) * (4 - len(sv))
        ll_fz_convert_color2(ss, *sv2, ds, dv, is_, params)
    else:
        ll_fz_convert_color2(ss, sv, ds, dv, is_, params)
    return (dv.v0, dv.v1, dv.v2, dv.v3)