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
def get_svg_image(self, matrix=None, text_as_path=1):
    """Make SVG image from page."""
    CheckParent(self)
    mediabox = mupdf.fz_bound_page(self.this)
    ctm = JM_matrix_from_py(matrix)
    tbounds = mediabox
    text_option = mupdf.FZ_SVG_TEXT_AS_PATH if text_as_path == 1 else mupdf.FZ_SVG_TEXT_AS_TEXT
    tbounds = mupdf.fz_transform_rect(tbounds, ctm)
    res = mupdf.fz_new_buffer(1024)
    out = mupdf.FzOutput(res)
    dev = mupdf.fz_new_svg_device(out, tbounds.x1 - tbounds.x0, tbounds.y1 - tbounds.y0, text_option, 1)
    mupdf.fz_run_page(self.this, dev, ctm, mupdf.FzCookie())
    mupdf.fz_close_device(dev)
    out.fz_close_output()
    text = JM_EscapeStrFromBuffer(res)
    return text