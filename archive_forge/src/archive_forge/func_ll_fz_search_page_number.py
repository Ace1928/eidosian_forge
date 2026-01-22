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
def ll_fz_search_page_number(doc, number, needle, hit_bbox, hit_max):
    """
    Wrapper for out-params of fz_search_page_number().
    Returns: int, int hit_mark
    """
    outparams = ll_fz_search_page_number_outparams()
    ret = ll_fz_search_page_number_outparams_fn(doc, number, needle, hit_bbox, hit_max, outparams)
    return (ret, outparams.hit_mark)