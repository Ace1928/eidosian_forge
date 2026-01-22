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
def ll_fz_error_callback():
    """
    Wrapper for out-params of fz_error_callback().
    Returns: fz_error_cb *, void *user
    """
    outparams = ll_fz_error_callback_outparams()
    ret = ll_fz_error_callback_outparams_fn(outparams)
    return (ret, outparams.user)