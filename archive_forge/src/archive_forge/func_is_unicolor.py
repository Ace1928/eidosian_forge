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
def is_unicolor(self):
    """
        Check if pixmap has only one color.
        """
    pm = self.this
    n = pm.n()
    count = pm.w() * pm.h() * n

    def _pixmap_read_samples(pm, offset, n):
        ret = list()
        for i in range(n):
            ret.append(mupdf.fz_samples_get(pm, offset + i))
        return ret
    sample0 = _pixmap_read_samples(pm, 0, n)
    for offset in range(n, count, n):
        sample = _pixmap_read_samples(pm, offset, n)
        if sample != sample0:
            return False
    return True