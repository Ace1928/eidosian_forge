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
def ll_fz_clamp_color(cs, in_):
    """
    Wrapper for out-params of fz_clamp_color().
    Returns: float out
    """
    outparams = ll_fz_clamp_color_outparams()
    ret = ll_fz_clamp_color_outparams_fn(cs, in_, outparams)
    return outparams.out