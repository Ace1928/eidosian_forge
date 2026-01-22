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
def ll_pdf_new_encrypt(opwd_utf8, upwd_utf8, id, permissions, algorithm):
    """ Low-level wrapper for `::pdf_new_encrypt()`."""
    return _mupdf.ll_pdf_new_encrypt(opwd_utf8, upwd_utf8, id, permissions, algorithm)