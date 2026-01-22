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
def ll_fz_new_deflated_data(source, source_length, level):
    """
    Wrapper for out-params of fz_new_deflated_data().
    Returns: unsigned char *, size_t compressed_length
    """
    outparams = ll_fz_new_deflated_data_outparams()
    ret = ll_fz_new_deflated_data_outparams_fn(source, source_length, level, outparams)
    return (ret, outparams.compressed_length)