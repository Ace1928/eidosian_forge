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
def jm_lineart_fill_path(dev, ctx, path, even_odd, ctm, colorspace, color, alpha, color_params):
    even_odd = True if even_odd else False
    try:
        assert isinstance(ctm, mupdf.fz_matrix)
        dev.ctm = mupdf.FzMatrix(ctm)
        dev.path_type = trace_device_FILL_PATH
        jm_lineart_path(dev, ctx, path)
        if dev.pathdict is None:
            return
        dev.pathdict[dictkey_type] = 'f'
        dev.pathdict['even_odd'] = even_odd
        dev.pathdict['fill_opacity'] = alpha
        dev.pathdict['fill'] = jm_lineart_color(colorspace, color)
        dev.pathdict[dictkey_rect] = JM_py_from_rect(dev.pathrect)
        dev.pathdict['seqno'] = dev.seqno
        dev.pathdict['layer'] = dev.layer_name
        if dev.clips:
            dev.pathdict['level'] = dev.depth
        jm_append_merge(dev)
        dev.seqno += 1
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise