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
def ll_fz_new_display_list_from_svg(buf, base_uri, dir):
    """
    Wrapper for out-params of fz_new_display_list_from_svg().
    Returns: fz_display_list *, float w, float h
    """
    outparams = ll_fz_new_display_list_from_svg_outparams()
    ret = ll_fz_new_display_list_from_svg_outparams_fn(buf, base_uri, dir, outparams)
    return (ret, outparams.w, outparams.h)