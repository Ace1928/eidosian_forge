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
def jm_bbox_fill_path(dev, ctx, path, even_odd, ctm, colorspace, color, alpha, color_params):
    even_odd = True if even_odd else False
    try:
        jm_bbox_add_rect(dev, ctx, mupdf.ll_fz_bound_path(path, None, ctm), 'fill-path')
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise