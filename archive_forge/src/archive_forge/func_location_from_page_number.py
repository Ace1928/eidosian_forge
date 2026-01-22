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
def location_from_page_number(self, pno):
    """Convert pno to (chapter, page)."""
    if self.is_closed:
        raise ValueError('document closed')
    this_doc = self.this
    loc = mupdf.fz_make_location(-1, -1)
    page_count = mupdf.fz_count_pages(this_doc)
    while pno < 0:
        pno += page_count
    if pno >= page_count:
        raise ValueError(MSG_BAD_PAGENO)
    loc = mupdf.fz_location_from_page_number(this_doc, pno)
    return (loc.chapter, loc.page)