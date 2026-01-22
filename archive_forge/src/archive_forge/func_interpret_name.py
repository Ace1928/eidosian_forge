from __future__ import annotations
import calendar
import codecs
import collections
import mmap
import os
import re
import time
import zlib
@classmethod
def interpret_name(cls, raw, as_text=False):
    name = b''
    for m in cls.re_hashes_in_name.finditer(raw):
        if m.group(3):
            name += m.group(1) + bytearray.fromhex(m.group(3).decode('us-ascii'))
        else:
            name += m.group(1)
    if as_text:
        return name.decode('utf-8')
    else:
        return bytes(name)