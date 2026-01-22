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
def reload_page(self, page: Page) -> Page:
    """Make a fresh copy of a page."""
    old_annots = {}
    pno = page.number
    for k, v in page._annot_refs.items():
        old_annots[k] = v
    refs_old = page.this.m_internal.refs
    m_internal_old = page.this.m_internal_value()
    page.this = None
    page._erase()
    page = None
    TOOLS.store_shrink(100)
    page = self.load_page(pno)
    for k, v in old_annots.items():
        annot = old_annots[k]
        page._annot_refs[k] = annot
    if refs_old == 1:
        pass
    else:
        m_internal_new = page.this.m_internal_value()
        assert m_internal_new != m_internal_old, f'refs_old={refs_old!r} m_internal_old={m_internal_old:#x} m_internal_new={m_internal_new:#x}'
    return page