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
def ll_fz_bitmap_details(bitmap):
    """
    Wrapper for out-params of fz_bitmap_details().
    Returns: int w, int h, int n, int stride
    """
    outparams = ll_fz_bitmap_details_outparams()
    ret = ll_fz_bitmap_details_outparams_fn(bitmap, outparams)
    return (outparams.w, outparams.h, outparams.n, outparams.stride)