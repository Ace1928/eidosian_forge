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
def util_measure_string(text, fontname, fontsize, encoding):
    font = mupdf.fz_new_base14_font(fontname)
    w = 0
    pos = 0
    while pos < len(text):
        t, c = mupdf.fz_chartorune(text[pos:])
        pos += t
        if encoding == mupdf.PDF_SIMPLE_ENCODING_GREEK:
            c = mupdf.fz_iso8859_7_from_unicode(c)
        elif encoding == mupdf.PDF_SIMPLE_ENCODING_CYRILLIC:
            c = mupdf.fz_windows_1251_from_unicode(c)
        else:
            c = mupdf.fz_windows_1252_from_unicode(c)
        if c < 0:
            c = 183
        g = mupdf.fz_encode_character(font, c)
        dw = mupdf.fz_advance_glyph(font, g, 0)
        w += dw
    ret = w * fontsize
    return ret