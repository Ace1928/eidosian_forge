import atexit
import binascii
import collections
import glob
import inspect
import io
import math
import os
import pathlib
import re
import string
import sys
import tarfile
import typing
import warnings
import weakref
import zipfile
from . import extra
from . import _extra
from . import utils
from .table import find_tables
def need_appearances(self, value=None):
    """Get/set the NeedAppearances value."""
    if not self.is_form_pdf:
        return None
    pdf = _as_pdf_document(self)
    oldval = -1
    appkey = 'NeedAppearances'
    form = mupdf.pdf_dict_getp(mupdf.pdf_trailer(pdf), 'Root/AcroForm')
    app = mupdf.pdf_dict_gets(form, appkey)
    if mupdf.pdf_is_bool(app):
        oldval = mupdf.pdf_to_bool(app)
    if value:
        mupdf.pdf_dict_puts(form, appkey, mupdf.PDF_TRUE)
    else:
        mupdf.pdf_dict_puts(form, appkey, mupdf.PDF_FALSE)
    if value is None:
        return oldval >= 0
    return value