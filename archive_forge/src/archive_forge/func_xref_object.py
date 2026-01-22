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
def xref_object(self, xref, compressed=0, ascii=0):
    """Get xref object source as a string."""
    if self.is_closed:
        raise ValueError('document closed')
    if g_use_extra:
        ret = extra.xref_object(self.this, xref, compressed, ascii)
        return ret
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    xreflen = mupdf.pdf_xref_len(pdf)
    if not _INRANGE(xref, 1, xreflen - 1) and xref != -1:
        raise ValueError(MSG_BAD_XREF)
    if xref > 0:
        obj = mupdf.pdf_load_object(pdf, xref)
    else:
        obj = mupdf.pdf_trailer(pdf)
    res = JM_object_to_buffer(mupdf.pdf_resolve_indirect(obj), compressed, ascii)
    text = JM_EscapeStrFromBuffer(res)
    return text