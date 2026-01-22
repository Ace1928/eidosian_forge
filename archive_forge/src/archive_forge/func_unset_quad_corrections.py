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
def unset_quad_corrections(on=None):
    """
        Set ascender / descender corrections on or off.
        """
    global g_skip_quad_corrections
    if on is not None:
        g_skip_quad_corrections = bool(on)
    return g_skip_quad_corrections