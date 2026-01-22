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
def get_outline_xrefs(self):
    """Get list of outline xref numbers."""
    xrefs = []
    pdf = _as_pdf_document(self)
    if not pdf:
        return xrefs
    root = mupdf.pdf_dict_get(mupdf.pdf_trailer(pdf), PDF_NAME('Root'))
    if not root.m_internal:
        return xrefs
    olroot = mupdf.pdf_dict_get(root, PDF_NAME('Outlines'))
    if not olroot.m_internal:
        return xrefs
    first = mupdf.pdf_dict_get(olroot, PDF_NAME('First'))
    if not first.m_internal:
        return xrefs
    xrefs = JM_outline_xrefs(first, xrefs)
    return xrefs