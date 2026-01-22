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
def pdf_field_event_validate_outparams_fn(doc, field, value):
    """
    Class-aware helper for out-params of pdf_field_event_validate() [pdf_field_event_validate()].
    """
    ret, newvalue = ll_pdf_field_event_validate(doc.m_internal, field.m_internal, value)
    return (ret, newvalue)