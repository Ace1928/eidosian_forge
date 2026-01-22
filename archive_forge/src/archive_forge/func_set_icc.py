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
@staticmethod
def set_icc(on=0):
    """Set ICC color handling on or off."""
    if on:
        if mupdf.FZ_ENABLE_ICC:
            mupdf.fz_enable_icc()
        else:
            RAISEPY('MuPDF built w/o ICC support', PyExc_ValueError)
    elif mupdf.FZ_ENABLE_ICC:
        mupdf.fz_disable_icc()