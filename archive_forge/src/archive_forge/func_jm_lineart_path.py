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
def jm_lineart_path(dev, ctx, path):
    """
    Create the "items" list of the path dictionary
    * either create or empty the path dictionary
    * reset the end point of the path
    * reset count of consecutive lines
    * invoke fz_walk_path(), which create the single items
    * if no items detected, empty path dict again
    """
    try:
        dev.pathrect = mupdf.FzRect(mupdf.FzRect.Fixed_INFINITE)
        dev.linecount = 0
        dev.lastpoint = mupdf.FzPoint(0, 0)
        dev.pathdict = dict()
        dev.pathdict[dictkey_items] = []
        walker = Walker(dev)
        mupdf.fz_walk_path(mupdf.FzPath(mupdf.ll_fz_keep_path(path)), walker, walker.m_internal)
        if not dev.pathdict[dictkey_items]:
            dev.pathdict = None
    except Exception:
        if g_exceptions_verbose:
            exception_info()
        raise