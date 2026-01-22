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
def insert_font(self, fontname='helv', fontfile=None, fontbuffer=None, set_simple=False, wmode=0, encoding=0):
    doc = self.parent
    if doc is None:
        raise ValueError('orphaned object: parent is None')
    idx = 0
    if fontname.startswith('/'):
        fontname = fontname[1:]
    inv_chars = INVALID_NAME_CHARS.intersection(fontname)
    if inv_chars != set():
        raise ValueError(f'bad fontname chars {inv_chars}')
    font = CheckFont(self, fontname)
    if font is not None:
        xref = font[0]
        if CheckFontInfo(doc, xref):
            return xref
        doc.get_char_widths(xref)
        return xref
    bfname = Base14_fontdict.get(fontname.lower(), None)
    serif = 0
    CJK_number = -1
    CJK_list_n = ['china-t', 'china-s', 'japan', 'korea']
    CJK_list_s = ['china-ts', 'china-ss', 'japan-s', 'korea-s']
    try:
        CJK_number = CJK_list_n.index(fontname)
        serif = 0
    except Exception:
        if g_exceptions_verbose > 1:
            exception_info()
        pass
    if CJK_number < 0:
        try:
            CJK_number = CJK_list_s.index(fontname)
            serif = 1
        except Exception:
            if g_exceptions_verbose > 1:
                exception_info()
            pass
    if fontname.lower() in fitz_fontdescriptors.keys():
        import pymupdf_fonts
        fontbuffer = pymupdf_fonts.myfont(fontname)
        del pymupdf_fonts
    if fontfile is not None:
        if type(fontfile) is str:
            fontfile_str = fontfile
        elif hasattr(fontfile, 'absolute'):
            fontfile_str = str(fontfile)
        elif hasattr(fontfile, 'name'):
            fontfile_str = fontfile.name
        else:
            raise ValueError('bad fontfile')
    else:
        fontfile_str = None
    val = self._insertFont(fontname, bfname, fontfile_str, fontbuffer, set_simple, idx, wmode, serif, encoding, CJK_number)
    if not val:
        return val
    xref = val[0]
    fontdict = val[1]
    if CheckFontInfo(doc, xref):
        return xref
    doc.get_char_widths(xref, fontdict=fontdict)
    return xref