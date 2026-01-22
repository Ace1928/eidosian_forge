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
def ll_pdf_page_presentation(page, transition):
    """
    Wrapper for out-params of pdf_page_presentation().
    Returns: fz_transition *, float duration
    """
    outparams = ll_pdf_page_presentation_outparams()
    ret = ll_pdf_page_presentation_outparams_fn(page, transition, outparams)
    return (ret, outparams.duration)