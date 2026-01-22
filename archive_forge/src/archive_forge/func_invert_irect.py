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
def invert_irect(self, bbox=None):
    """Invert the colors inside a bbox."""
    pm = self.this
    if not mupdf.fz_pixmap_colorspace(pm):
        JM_Warning('ignored for stencil pixmap')
        return False
    r = JM_irect_from_py(bbox)
    if mupdf.fz_is_infinite_irect(r):
        r = mupdf.fz_pixmap_bbox(pm)
    return bool(JM_invert_pixmap_rect(pm, r))