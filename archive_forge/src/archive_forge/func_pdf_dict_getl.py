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
def pdf_dict_getl(obj, *tail):
    """
    Python implementation of pdf_dict_getl(), because SWIG
    doesn't handle variadic args. Each item in `tail` should be
    a `mupdf.PdfObj`.
    """
    for key in tail:
        if not obj.m_internal:
            break
        obj = pdf_dict_get(obj, key)
    assert isinstance(obj, PdfObj)
    return obj