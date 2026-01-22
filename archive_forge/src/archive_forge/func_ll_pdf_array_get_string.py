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
def ll_pdf_array_get_string(array, index):
    """
    Wrapper for out-params of pdf_array_get_string().
    Returns: const char *, size_t sizep
    """
    outparams = ll_pdf_array_get_string_outparams()
    ret = ll_pdf_array_get_string_outparams_fn(array, index, outparams)
    return (ret, outparams.sizep)