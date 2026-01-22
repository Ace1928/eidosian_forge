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
def prev_location(self, page_id):
    """Get (chapter, page) of previous page."""
    if self.is_closed or self.is_encrypted:
        raise ValueError('document closed or encrypted')
    if type(page_id) is int:
        page_id = (0, page_id)
    if page_id not in self:
        raise ValueError('page id not in document')
    if page_id == (0, 0):
        return ()
    chapter, pno = page_id
    loc = mupdf.fz_make_location(chapter, pno)
    prev_loc = mupdf.fz_previous_page(self.this, loc)
    return (prev_loc.chapter, prev_loc.page)