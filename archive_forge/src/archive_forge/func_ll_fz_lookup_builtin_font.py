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
def ll_fz_lookup_builtin_font(name, bold, italic):
    """
    Wrapper for out-params of fz_lookup_builtin_font().
    Returns: const unsigned char *, int len
    """
    outparams = ll_fz_lookup_builtin_font_outparams()
    ret = ll_fz_lookup_builtin_font_outparams_fn(name, bold, italic, outparams)
    return (ret, outparams.len)