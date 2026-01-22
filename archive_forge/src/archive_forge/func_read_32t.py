from __future__ import annotations
import io
import os
import struct
import sys
from . import Image, ImageFile, PngImagePlugin, features
def read_32t(fobj, start_length, size):
    start, length = start_length
    fobj.seek(start)
    sig = fobj.read(4)
    if sig != b'\x00\x00\x00\x00':
        msg = 'Unknown signature, expecting 0x00000000'
        raise SyntaxError(msg)
    return read_32(fobj, (start + 4, length - 4), size)