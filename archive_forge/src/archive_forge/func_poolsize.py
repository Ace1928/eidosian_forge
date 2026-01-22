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
def poolsize(self):
    """TextPage current poolsize."""
    tpage = self.this
    pool = mupdf.Pool(tpage.m_internal.pool)
    size = mupdf.fz_pool_size(pool)
    pool.m_internal = None
    return size