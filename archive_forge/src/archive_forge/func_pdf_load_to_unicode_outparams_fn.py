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
def pdf_load_to_unicode_outparams_fn(doc, font, collection, cmapstm):
    """
    Class-aware helper for out-params of pdf_load_to_unicode() [pdf_load_to_unicode()].
    """
    strings = ll_pdf_load_to_unicode(doc.m_internal, font.m_internal, collection, cmapstm.m_internal)
    return strings