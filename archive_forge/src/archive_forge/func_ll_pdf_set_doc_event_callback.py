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
def ll_pdf_set_doc_event_callback(doc, event_cb, free_event_data_cb, data):
    """ Low-level wrapper for `::pdf_set_doc_event_callback()`."""
    return _mupdf.ll_pdf_set_doc_event_callback(doc, event_cb, free_event_data_cb, data)