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
def set_rotation(self, rotation):
    """Set page rotation."""
    CheckParent(self)
    page = mupdf.pdf_page_from_fz_page(self.this)
    ASSERT_PDF(page)
    rot = JM_norm_rotation(rotation)
    mupdf.pdf_dict_put_int(page.obj(), PDF_NAME('Rotate'), rot)