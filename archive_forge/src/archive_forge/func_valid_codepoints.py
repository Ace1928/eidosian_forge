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
def valid_codepoints(self):
    """
        list of valid unicodes of a fz_font
        """
    return []
    from array import array
    gc = self.glyph_count
    cp = array('l', (0,) * gc)
    arr = cp.buffer_info()
    self._valid_unicodes(arr)
    return array('l', sorted(set(cp))[1:])