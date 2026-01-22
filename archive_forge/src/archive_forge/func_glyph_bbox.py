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
def glyph_bbox(self, char, language=None, script=0, small_caps=0):
    """Return the glyph bbox of a unicode (font size 1)."""
    lang = mupdf.fz_text_language_from_string(language)
    if small_caps:
        gid = mupdf.fz_encode_character_sc(self.this, char)
        if gid >= 0:
            font = self.this
    else:
        gid, font = mupdf.fz_encode_character_with_fallback(self.this, char, script, lang)
    return Rect(mupdf.fz_bound_glyph(font, gid, mupdf.FzMatrix()))