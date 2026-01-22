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
def pdf_undoredo_state_outparams_fn(doc):
    """
    Class-aware helper for out-params of pdf_undoredo_state() [pdf_undoredo_state()].
    """
    ret, steps = ll_pdf_undoredo_state(doc.m_internal)
    return (ret, steps)