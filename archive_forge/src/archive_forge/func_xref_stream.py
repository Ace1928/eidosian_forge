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
def xref_stream(self, xref):
    """Get decompressed xref stream."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    pdf = _as_pdf_document(self)
    ASSERT_PDF(pdf)
    xreflen = mupdf.pdf_xref_len(pdf)
    if not _INRANGE(xref, 1, xreflen - 1) and xref != -1:
        raise ValueError(MSG_BAD_XREF)
    if xref >= 0:
        obj = mupdf.pdf_new_indirect(pdf, xref, 0)
    else:
        obj = mupdf.pdf_trailer(pdf)
    r = None
    if mupdf.pdf_is_stream(obj):
        res = mupdf.pdf_load_stream_number(pdf, xref)
        r = JM_BinFromBuffer(res)
    return r