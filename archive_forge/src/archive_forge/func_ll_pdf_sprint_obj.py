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
def ll_pdf_sprint_obj(buf, cap, obj, tight, ascii):
    """
    Wrapper for out-params of pdf_sprint_obj().
    Returns: char *, size_t len
    """
    outparams = ll_pdf_sprint_obj_outparams()
    ret = ll_pdf_sprint_obj_outparams_fn(buf, cap, obj, tight, ascii, outparams)
    return (ret, outparams.len)