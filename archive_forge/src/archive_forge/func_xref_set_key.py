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
def xref_set_key(self, xref, key, value):
    """Set the value of a PDF dictionary key."""
    if self.is_closed:
        raise ValueError('document closed')
    if not key or not isinstance(key, str) or INVALID_NAME_CHARS.intersection(key) not in (set(), {'/'}):
        raise ValueError("bad 'key'")
    if not isinstance(value, str) or not value or (value[0] == '/' and INVALID_NAME_CHARS.intersection(value[1:]) != set()):
        raise ValueError("bad 'value'")
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    xreflen = mupdf.pdf_xref_len(pdf)
    if not _INRANGE(xref, 1, xreflen - 1) and xref != -1:
        raise ValueError(MSG_BAD_XREF)
    if xref != -1:
        obj = mupdf.pdf_load_object(pdf, xref)
    else:
        obj = mupdf.pdf_trailer(pdf)
    new_obj = JM_set_object_value(obj, key, value)
    if not new_obj.m_internal:
        return
    if xref != -1:
        mupdf.pdf_update_object(pdf, xref, new_obj)
    else:
        n = mupdf.pdf_dict_len(new_obj)
        for i in range(n):
            mupdf.pdf_dict_put(obj, mupdf.pdf_dict_get_key(new_obj, i), mupdf.pdf_dict_get_val(new_obj, i))