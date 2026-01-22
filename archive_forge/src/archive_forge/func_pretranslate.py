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
def pretranslate(self, tx, ty):
    """Calculate pre translation and replace current matrix."""
    tx = float(tx)
    ty = float(ty)
    self.e += tx * self.a + ty * self.c
    self.f += tx * self.b + ty * self.d
    return self