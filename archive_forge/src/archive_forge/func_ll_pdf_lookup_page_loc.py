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
def ll_pdf_lookup_page_loc(doc, needle):
    """
    Wrapper for out-params of pdf_lookup_page_loc().
    Returns: pdf_obj *, ::pdf_obj *parentp, int indexp
    """
    outparams = ll_pdf_lookup_page_loc_outparams()
    ret = ll_pdf_lookup_page_loc_outparams_fn(doc, needle, outparams)
    return (ret, outparams.parentp, outparams.indexp)