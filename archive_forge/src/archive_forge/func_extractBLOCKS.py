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
def extractBLOCKS(self):
    """Return a list with text block information."""
    if g_use_extra:
        return extra.extractBLOCKS(self.this)
    block_n = -1
    this_tpage = self.this
    tp_rect = mupdf.FzRect(this_tpage.m_internal.mediabox)
    res = mupdf.fz_new_buffer(1024)
    lines = []
    for block in this_tpage:
        block_n += 1
        blockrect = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)
        if block.m_internal.type == mupdf.FZ_STEXT_BLOCK_TEXT:
            mupdf.fz_clear_buffer(res)
            line_n = -1
            last_char = 0
            for line in block:
                line_n += 1
                linerect = mupdf.FzRect(mupdf.FzRect.Fixed_EMPTY)
                for ch in line:
                    cbbox = JM_char_bbox(line, ch)
                    if not JM_rects_overlap(tp_rect, cbbox) and (not mupdf.fz_is_infinite_rect(tp_rect)):
                        continue
                    JM_append_rune(res, ch.m_internal.c)
                    last_char = ch.m_internal.c
                    linerect = mupdf.fz_union_rect(linerect, cbbox)
                if last_char != 10 and (not mupdf.fz_is_empty_rect(linerect)):
                    mupdf.fz_append_byte(res, 10)
                blockrect = mupdf.fz_union_rect(blockrect, linerect)
            text = JM_EscapeStrFromBuffer(res)
        elif JM_rects_overlap(tp_rect, block.m_internal.bbox) or mupdf.fz_is_infinite_rect(tp_rect):
            img = block.i_image()
            cs = img.colorspace()
            text = '<image: %s, width: %d, height: %d, bpc: %d>' % (mupdf.fz_colorspace_name(cs), img.w(), img.h(), img.bpc())
            blockrect = mupdf.fz_union_rect(blockrect, mupdf.FzRect(block.m_internal.bbox))
        if not mupdf.fz_is_empty_rect(blockrect):
            litem = (blockrect.x0, blockrect.y0, blockrect.x1, blockrect.y1, text, block_n, block.m_internal.type)
            lines.append(litem)
    return lines