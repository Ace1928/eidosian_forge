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
def ll_pdf_annot_default_appearance(annot, color):
    """
    Wrapper for out-params of pdf_annot_default_appearance().
    Returns: const char *font, float size, int n
    """
    outparams = ll_pdf_annot_default_appearance_outparams()
    ret = ll_pdf_annot_default_appearance_outparams_fn(annot, color, outparams)
    return (outparams.font, outparams.size, outparams.n)