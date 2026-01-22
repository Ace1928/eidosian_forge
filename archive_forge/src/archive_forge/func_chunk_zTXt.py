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
def chunk_zTXt(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    try:
        k, v = s.split(b'\x00', 1)
    except ValueError:
        k = s
        v = b''
    if v:
        comp_method = v[0]
    else:
        comp_method = 0
    if comp_method != 0:
        msg = f'Unknown compression method {comp_method} in zTXt chunk'
        raise SyntaxError(msg)
    try:
        v = _safe_zlib_decompress(v[1:])
    except ValueError:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            v = b''
        else:
            raise
    except zlib.error:
        v = b''
    if k:
        k = k.decode('latin-1', 'strict')
        v = v.decode('latin-1', 'replace')
        self.im_info[k] = self.im_text[k] = v
        self.check_text_memory(len(v))
    return s