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
def move_page(self, pno: int, to: int=-1):
    """Move a page within a PDF document.

        Args:
            pno: source page number.
            to: put before this page, '-1' means after last page.
        """
    if self.is_closed:
        raise ValueError('document closed')
    page_count = len(self)
    if pno not in range(page_count) or to not in range(-1, page_count):
        raise ValueError('bad page number(s)')
    before = 1
    copy = 0
    if to == -1:
        to = page_count - 1
        before = 0
    return self._move_copy_page(pno, to, before, copy)