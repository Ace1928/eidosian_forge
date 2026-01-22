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
def ll_fz_lookup_cjk_font(ordering):
    """
    Wrapper for out-params of fz_lookup_cjk_font().
    Returns: const unsigned char *, int len, int index
    """
    outparams = ll_fz_lookup_cjk_font_outparams()
    ret = ll_fz_lookup_cjk_font_outparams_fn(ordering, outparams)
    return (ret, outparams.len, outparams.index)