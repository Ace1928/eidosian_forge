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
def has_glyph(self, chr, language=None, script=0, fallback=0, small_caps=0):
    """Check whether font has a glyph for this unicode."""
    if fallback:
        lang = mupdf.fz_text_language_from_string(language)
        gid, font = mupdf.fz_encode_character_with_fallback(self.this, chr, script, lang)
    elif small_caps:
        gid = mupdf.fz_encode_character_sc(self.this, chr)
    else:
        gid = mupdf.fz_encode_character(self.this, chr)
    return gid