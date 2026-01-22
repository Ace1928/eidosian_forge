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
def jm_lineart_color(colorspace, color):
    if colorspace:
        try:
            cs = mupdf.FzColorspace(mupdf.FzColorspace.Fixed_RGB)
            cp = mupdf.FzColorParams()
            rgb = mupdf.ll_fz_convert_color(colorspace, color, cs.m_internal, None, cp.internal())
        except Exception:
            if g_exceptions_verbose:
                exception_info()
            raise
        return rgb[:3]
    return ()