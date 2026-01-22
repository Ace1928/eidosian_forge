from __future__ import annotations
import itertools
import logging
import re
import struct
import warnings
import zlib
from enum import IntEnum
from . import Image, ImageChops, ImageFile, ImagePalette, ImageSequence
from ._binary import i16be as i16
from ._binary import i32be as i32
from ._binary import o8
from ._binary import o16be as o16
from ._binary import o32be as o32
def load_read(self, read_bytes):
    """internal: read more image data"""
    while self.__idat == 0:
        self.fp.read(4)
        cid, pos, length = self.png.read()
        if cid not in [b'IDAT', b'DDAT', b'fdAT']:
            self.png.push(cid, pos, length)
            return b''
        if cid == b'fdAT':
            try:
                self.png.call(cid, pos, length)
            except EOFError:
                pass
            self.__idat = length - 4
        else:
            self.__idat = length
    if read_bytes <= 0:
        read_bytes = self.__idat
    else:
        read_bytes = min(read_bytes, self.__idat)
    self.__idat = self.__idat - read_bytes
    return self.fp.read(read_bytes)