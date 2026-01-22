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
def tint_with(self, black, white):
    """Tint colors with modifiers for black and white."""
    if not self.colorspace or self.colorspace.n > 3:
        message('warning: colorspace invalid for function')
        return
    return mupdf.fz_tint_pixmap(self.this, black, white)