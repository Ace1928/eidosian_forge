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
def load_links(self):
    """Get first Link."""
    CheckParent(self)
    val = mupdf.fz_load_links(self.this)
    if not val.m_internal:
        return
    val = Link(val)
    val.thisown = True
    val.parent = weakref.proxy(self)
    self._annot_refs[id(val)] = val
    val.xref = 0
    val.id = ''
    if self.parent.is_pdf:
        xrefs = self.annot_xrefs()
        xrefs = [x for x in xrefs if x[1] == mupdf.PDF_ANNOT_LINK]
        if xrefs:
            link_id = xrefs[0]
            val.xref = link_id[0]
            val.id = link_id[2]
    else:
        val.xref = 0
        val.id = ''
    return val