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
def prescale(self, sx, sy):
    """Calculate pre scaling and replace current matrix."""
    sx = float(sx)
    sy = float(sy)
    self.a *= sx
    self.b *= sx
    self.c *= sy
    self.d *= sy
    return self