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
def ll_fz_grisu(f, s):
    """
    Wrapper for out-params of fz_grisu().
    Returns: int, int exp
    """
    outparams = ll_fz_grisu_outparams()
    ret = ll_fz_grisu_outparams_fn(f, s, outparams)
    return (ret, outparams.exp)