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
@property
def first_annot(self):
    """First annotation."""
    CheckParent(self)
    page = self._pdf_page()
    if not page:
        return
    annot = mupdf.pdf_first_annot(page)
    if not annot.m_internal:
        return
    val = Annot(annot)
    val.thisown = True
    val.parent = weakref.proxy(self)
    self._annot_refs[id(val)] = val
    return val