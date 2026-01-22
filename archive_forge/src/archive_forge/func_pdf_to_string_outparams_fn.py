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
def pdf_to_string_outparams_fn(obj):
    """
    Class-aware helper for out-params of pdf_to_string() [pdf_to_string()].
    """
    ret, sizep = ll_pdf_to_string(obj.m_internal)
    return (ret, sizep)