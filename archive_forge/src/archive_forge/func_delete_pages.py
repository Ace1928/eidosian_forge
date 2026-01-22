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
def delete_pages(self, *args, **kw):
    """Delete pages from a PDF.

        Args:
            Either keywords 'from_page'/'to_page', or two integers to
            specify the first/last page to delete.
            Or a list/tuple/range object, which can contain arbitrary
            page numbers.
        """
    if not self.is_pdf:
        raise ValueError('is no PDF')
    if self.is_closed:
        raise ValueError('document closed')
    page_count = self.page_count
    f = t = -1
    if kw:
        if args:
            raise ValueError('cannot mix keyword and positional argument')
        f = kw.get('from_page', -1)
        t = kw.get('to_page', -1)
        while f < 0:
            f += page_count
        while t < 0:
            t += page_count
        if not f <= t < page_count:
            raise ValueError('bad page number(s)')
        numbers = tuple(range(f, t + 1))
    else:
        if len(args) > 2 or args == []:
            raise ValueError('need 1 or 2 positional arguments')
        if len(args) == 2:
            f, t = args
            if not (type(f) is int and type(t) is int):
                raise ValueError('both arguments must be int')
            if f > t:
                f, t = (t, f)
            if not f <= t < page_count:
                raise ValueError('bad page number(s)')
            numbers = tuple(range(f, t + 1))
        else:
            r = args[0]
            if type(r) not in (int, range, list, tuple):
                raise ValueError('need int or sequence if one argument')
            numbers = tuple(r)
    numbers = list(map(int, set(numbers)))
    if numbers == []:
        message('nothing to delete')
        return
    numbers.sort()
    if numbers[0] < 0 or numbers[-1] >= page_count:
        raise ValueError('bad page number(s)')
    frozen_numbers = frozenset(numbers)
    toc = self.get_toc()
    for i, xref in enumerate(self.get_outline_xrefs()):
        if toc[i][2] - 1 in frozen_numbers:
            self._remove_toc_item(xref)
    self._remove_links_to(frozen_numbers)
    for i in reversed(numbers):
        self._delete_page(i)
    self._reset_page_refs()