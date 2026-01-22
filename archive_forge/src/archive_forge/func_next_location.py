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
def next_location(self, page_id):
    """Get (chapter, page) of next page."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    if type(page_id) is int:
        page_id = (0, page_id)
    if page_id not in self:
        raise ValueError('page id not in document')
    if tuple(page_id) == self.last_location:
        return ()
    this_doc = _as_fz_document(self)
    val = page_id[0]
    if not isinstance(val, int):
        RAISEPY(MSG_BAD_PAGEID, PyExc_ValueError)
    chapter = val
    val = page_id[1]
    pno = val
    loc = mupdf.fz_make_location(chapter, pno)
    next_loc = mupdf.fz_next_page(this_doc, loc)
    return (next_loc.chapter, next_loc.page)