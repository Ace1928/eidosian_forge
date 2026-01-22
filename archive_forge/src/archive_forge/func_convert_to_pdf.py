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
def convert_to_pdf(self, from_page=0, to_page=-1, rotate=0):
    """Convert document to a PDF, selecting page range and optional rotation. Output bytes object."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    fz_doc = self.this
    fp = from_page
    tp = to_page
    srcCount = mupdf.fz_count_pages(fz_doc)
    if fp < 0:
        fp = 0
    if fp > srcCount - 1:
        fp = srcCount - 1
    if tp < 0:
        tp = srcCount - 1
    if tp > srcCount - 1:
        tp = srcCount - 1
    len0 = len(JM_mupdf_warnings_store)
    doc = JM_convert_to_pdf(fz_doc, fp, tp, rotate)
    len1 = len(JM_mupdf_warnings_store)
    for i in range(len0, len1):
        PySys_WriteStderr(f'{JM_mupdf_warnings_store[i]}\n')
    return doc