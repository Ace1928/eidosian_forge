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
def glyph_cache_empty():
    """
        Empty the glyph cache.
        """
    mupdf.fz_purge_glyph_cache()