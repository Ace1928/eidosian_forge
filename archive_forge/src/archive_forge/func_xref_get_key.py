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
def xref_get_key(self, xref, key):
    """Get PDF dict key value of object at 'xref'."""
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    xreflen = mupdf.pdf_xref_len(pdf)
    if not _INRANGE(xref, 1, xreflen - 1) and xref != -1:
        raise ValueError(MSG_BAD_XREF)
    if xref > 0:
        obj = mupdf.pdf_load_object(pdf, xref)
    else:
        obj = mupdf.pdf_trailer(pdf)
    if not obj.m_internal:
        return ('null', 'null')
    subobj = mupdf.pdf_dict_getp(obj, key)
    if not subobj.m_internal:
        return ('null', 'null')
    text = None
    if mupdf.pdf_is_indirect(subobj):
        type = 'xref'
        text = '%i 0 R' % mupdf.pdf_to_num(subobj)
    elif mupdf.pdf_is_array(subobj):
        type = 'array'
    elif mupdf.pdf_is_dict(subobj):
        type = 'dict'
    elif mupdf.pdf_is_int(subobj):
        type = 'int'
        text = '%i' % mupdf.pdf_to_int(subobj)
    elif mupdf.pdf_is_real(subobj):
        type = 'float'
    elif mupdf.pdf_is_null(subobj):
        type = 'null'
        text = 'null'
    elif mupdf.pdf_is_bool(subobj):
        type = 'bool'
        if mupdf.pdf_to_bool(subobj):
            text = 'true'
        else:
            text = 'false'
    elif mupdf.pdf_is_name(subobj):
        type = 'name'
        text = '/%s' % mupdf.pdf_to_name(subobj)
    elif mupdf.pdf_is_string(subobj):
        type = 'string'
        text = JM_UnicodeFromStr(mupdf.pdf_to_text_string(subobj))
    else:
        type = 'unknown'
    if text is None:
        res = JM_object_to_buffer(subobj, 1, 0)
        text = JM_UnicodeFromBuffer(res)
    return (type, text)