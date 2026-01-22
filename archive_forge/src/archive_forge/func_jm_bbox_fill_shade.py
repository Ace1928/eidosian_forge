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
def jm_bbox_fill_shade(dev, ctx, shade, ctm, alpha, color_params):
    try:
        jm_bbox_add_rect(dev, ctx, mupdf.ll_fz_bound_shade(shade, ctm), 'fill-shade')
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise