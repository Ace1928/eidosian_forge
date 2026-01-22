from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
def decode_text(b):
    if b[:len(codecs.BOM_UTF16_BE)] == codecs.BOM_UTF16_BE:
        return b[len(codecs.BOM_UTF16_BE):].decode('utf_16_be')
    else:
        return ''.join((PDFDocEncoding.get(byte, chr(byte)) for byte in b))