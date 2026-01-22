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
def rect_delta(self):
    """
        annotation delta values to rectangle
        """
    annot_obj = mupdf.pdf_annot_obj(self.this)
    arr = mupdf.pdf_dict_get(annot_obj, PDF_NAME('RD'))
    if mupdf.pdf_array_len(arr) == 4:
        return (mupdf.pdf_to_real(mupdf.pdf_array_get(arr, 0)), mupdf.pdf_to_real(mupdf.pdf_array_get(arr, 1)), -mupdf.pdf_to_real(mupdf.pdf_array_get(arr, 2)), -mupdf.pdf_to_real(mupdf.pdf_array_get(arr, 3)))