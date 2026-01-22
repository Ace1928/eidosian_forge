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
def pdf_dict_putl(obj, val, *tail):
    """
    Python implementation of pdf_dict_putl(fz_context *ctx,
    pdf_obj *obj, pdf_obj *val, ...) because SWIG doesn't
    handle variadic args. Each item in `tail` should
    be a SWIG wrapper for a `PdfObj`.
    """
    if pdf_is_indirect(obj):
        obj = pdf_resolve_indirect_chain(obj)
    if not pdf_is_dict(obj):
        raise Exception(f'not a dict: {obj}')
    if not tail:
        return
    doc = pdf_get_bound_document(obj)
    for i, key in enumerate(tail[:-1]):
        assert isinstance(key, PdfObj), f'item {i} in `tail` should be a PdfObj but is a {type(key)}.'
        next_obj = pdf_dict_get(obj, key)
        if not next_obj.m_internal:
            next_obj = pdf_new_dict(doc, 1)
            pdf_dict_put(obj, key, next_obj)
        obj = next_obj
    key = tail[-1]
    pdf_dict_put(obj, key, val)