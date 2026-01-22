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
def line_ends(self):
    """Line end codes."""
    CheckParent(self)
    annot = self.this
    if not mupdf.pdf_annot_has_line_ending_styles(annot):
        return
    lstart = mupdf.pdf_annot_line_start_style(annot)
    lend = mupdf.pdf_annot_line_end_style(annot)
    return (lstart, lend)