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
def ll_fz_convert_error():
    """
    Wrapper for out-params of fz_convert_error().
    Returns: const char *, int code
    """
    outparams = ll_fz_convert_error_outparams()
    ret = ll_fz_convert_error_outparams_fn(outparams)
    return (ret, outparams.code)