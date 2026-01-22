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
@property
def bleedbox(self):
    """The BleedBox"""
    rect = self._other_box('BleedBox')
    if rect is None:
        return self.cropbox
    mb = self.mediabox
    return Rect(rect[0], mb.y1 - rect[3], rect[2], mb.y1 - rect[1])