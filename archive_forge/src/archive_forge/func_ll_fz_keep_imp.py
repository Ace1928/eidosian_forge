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
def ll_fz_keep_imp(p):
    """
    Wrapper for out-params of fz_keep_imp().
    Returns: void *, int refs
    """
    outparams = ll_fz_keep_imp_outparams()
    ret = ll_fz_keep_imp_outparams_fn(p, outparams)
    return (ret, outparams.refs)