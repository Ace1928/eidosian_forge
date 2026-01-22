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
def glyph_advance(self, chr_, language=None, script=0, wmode=0, small_caps=0):
    """Return the glyph width of a unicode (font size 1)."""
    lang = mupdf.fz_text_language_from_string(language)
    if small_caps:
        gid = mupdf.fz_encode_character_sc(self.this, chr_)
        if gid >= 0:
            font = self.this
    else:
        gid, font = mupdf.fz_encode_character_with_fallback(self.this, chr_, script, lang)
    return mupdf.fz_advance_glyph(font, gid, wmode)