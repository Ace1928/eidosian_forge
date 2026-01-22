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
def ll_pdf_dict_putl(obj, val, *tail):
    """
    Python implementation of ll_pdf_dict_putl() because SWIG
    doesn't handle variadic args. Each item in `tail` should
    be a SWIG wrapper for a `pdf_obj`.
    """
    if ll_pdf_is_indirect(obj):
        obj = ll_pdf_resolve_indirect_chain(obj)
    if not pdf_is_dict(obj):
        raise Exception(f'not a dict: {obj}')
    if not tail:
        return
    doc = ll_pdf_get_bound_document(obj)
    for i, key in enumerate(tail[:-1]):
        assert isinstance(key, PdfObj), f'Item {i} in `tail` should be a pdf_obj but is a {type(key)}.'
        next_obj = ll_pdf_dict_get(obj, key)
        if not next_obj:
            next_obj = ll_pdf_new_dict(doc, 1)
            ll_pdf_dict_put(obj, key, next_obj)
        obj = next_obj
    key = tail[-1]
    ll_pdf_dict_put(obj, key, val)