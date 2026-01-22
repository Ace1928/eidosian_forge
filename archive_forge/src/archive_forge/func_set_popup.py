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
def set_popup(self, rect):
    """
        Create annotation 'Popup' or update rectangle.
        """
    CheckParent(self)
    annot = self.this
    pdfpage = mupdf.pdf_annot_page(annot)
    rot = JM_rotate_page_matrix(pdfpage)
    r = mupdf.fz_transform_rect(JM_rect_from_py(rect), rot)
    mupdf.pdf_set_annot_popup(annot, r)