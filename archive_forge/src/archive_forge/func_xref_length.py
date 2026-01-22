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
def xref_length(self):
    """Get length of xref table."""
    xreflen = 0
    pdf = _as_pdf_document(self)
    if pdf:
        xreflen = mupdf.pdf_xref_len(pdf)
    return xreflen