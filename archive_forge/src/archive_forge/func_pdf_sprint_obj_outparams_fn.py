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
def pdf_sprint_obj_outparams_fn(buf, cap, obj, tight, ascii):
    """
    Class-aware helper for out-params of pdf_sprint_obj() [pdf_sprint_obj()].
    """
    ret, len = ll_pdf_sprint_obj(buf, cap, obj.m_internal, tight, ascii)
    return (ret, len)