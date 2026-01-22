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
def fz_search_page_number_outparams_fn(doc, number, needle, hit_bbox, hit_max):
    """
    Class-aware helper for out-params of fz_search_page_number() [fz_search_page_number()].
    """
    ret, hit_mark = ll_fz_search_page_number(doc.m_internal, number, needle, hit_bbox.internal(), hit_max)
    return (ret, hit_mark)