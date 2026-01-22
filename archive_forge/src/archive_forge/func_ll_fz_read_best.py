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
def ll_fz_read_best(stm, initial, worst_case):
    """
    Wrapper for out-params of fz_read_best().
    Returns: fz_buffer *, int truncated
    """
    outparams = ll_fz_read_best_outparams()
    ret = ll_fz_read_best_outparams_fn(stm, initial, worst_case, outparams)
    return (ret, outparams.truncated)