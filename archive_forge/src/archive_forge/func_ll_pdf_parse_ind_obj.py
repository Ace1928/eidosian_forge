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
def ll_pdf_parse_ind_obj(doc, f):
    """
    Wrapper for out-params of pdf_parse_ind_obj().
    Returns: pdf_obj *, int num, int gen, int64_t stm_ofs, int try_repair
    """
    outparams = ll_pdf_parse_ind_obj_outparams()
    ret = ll_pdf_parse_ind_obj_outparams_fn(doc, f, outparams)
    return (ret, outparams.num, outparams.gen, outparams.stm_ofs, outparams.try_repair)