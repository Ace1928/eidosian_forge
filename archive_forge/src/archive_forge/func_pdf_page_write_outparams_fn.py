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
def pdf_page_write_outparams_fn(doc, mediabox):
    """
    Class-aware helper for out-params of pdf_page_write() [pdf_page_write()].
    """
    ret, presources, pcontents = ll_pdf_page_write(doc.m_internal, mediabox.internal())
    return (FzDevice(ret), PdfObj(presources), FzBuffer(pcontents))