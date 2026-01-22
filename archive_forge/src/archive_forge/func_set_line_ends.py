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
def set_line_ends(self, start, end):
    """Set line end codes."""
    CheckParent(self)
    annot = self.this
    if mupdf.pdf_annot_has_line_ending_styles(annot):
        mupdf.pdf_set_annot_line_ending_styles(annot, start, end)
    else:
        JM_Warning('bad annot type for line ends')