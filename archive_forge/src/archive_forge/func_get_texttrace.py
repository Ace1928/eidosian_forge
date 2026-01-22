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
def get_texttrace(self):
    CheckParent(self)
    old_rotation = self.rotation
    if old_rotation != 0:
        self.set_rotation(0)
    page = self.this
    rc = []
    if g_use_extra:
        dev = extra.JM_new_texttrace_device(rc)
    else:
        dev = JM_new_texttrace_device(rc)
    prect = mupdf.fz_bound_page(page)
    dev.ptm = mupdf.FzMatrix(1, 0, 0, -1, 0, prect.y1)
    mupdf.fz_run_page(page, dev, mupdf.FzMatrix(), mupdf.FzCookie())
    mupdf.fz_close_device(dev)
    if old_rotation != 0:
        self.set_rotation(old_rotation)
    return rc