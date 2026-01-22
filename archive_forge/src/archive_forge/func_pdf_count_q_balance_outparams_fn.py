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
def pdf_count_q_balance_outparams_fn(doc, res, stm):
    """
    Class-aware helper for out-params of pdf_count_q_balance() [pdf_count_q_balance()].
    """
    underflow, overflow = ll_pdf_count_q_balance(doc.m_internal, res.m_internal, stm.m_internal)
    return (underflow, overflow)