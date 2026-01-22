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
def get_bboxlog(self, layers=None):
    CheckParent(self)
    old_rotation = self.rotation
    if old_rotation != 0:
        self.set_rotation(0)
    page = self.this
    rc = []
    inc_layers = True if layers else False
    dev = JM_new_bbox_device(rc, inc_layers)
    mupdf.fz_run_page(page, dev, mupdf.FzMatrix(), mupdf.FzCookie())
    mupdf.fz_close_device(dev)
    if old_rotation != 0:
        self.set_rotation(old_rotation)
    return rc