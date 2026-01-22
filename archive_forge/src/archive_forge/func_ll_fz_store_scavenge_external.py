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
def ll_fz_store_scavenge_external(size):
    """
    Wrapper for out-params of fz_store_scavenge_external().
    Returns: int, int phase
    """
    outparams = ll_fz_store_scavenge_external_outparams()
    ret = ll_fz_store_scavenge_external_outparams_fn(size, outparams)
    return (ret, outparams.phase)