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
def ll_fz_chartorune(str):
    """
    Wrapper for out-params of fz_chartorune().
    Returns: int, int rune
    """
    outparams = ll_fz_chartorune_outparams()
    ret = ll_fz_chartorune_outparams_fn(str, outparams)
    return (ret, outparams.rune)