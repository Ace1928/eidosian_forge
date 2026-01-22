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
def set_open(self, is_open):
    """Set 'open' status of annotation or its Popup."""
    CheckParent(self)
    annot = self.this
    mupdf.pdf_set_annot_is_open(annot, is_open)