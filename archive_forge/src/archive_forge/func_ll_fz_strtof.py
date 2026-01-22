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
def ll_fz_strtof(s):
    """
    Wrapper for out-params of fz_strtof().
    Returns: float, char *es
    """
    outparams = ll_fz_strtof_outparams()
    ret = ll_fz_strtof_outparams_fn(s, outparams)
    return (ret, outparams.es)