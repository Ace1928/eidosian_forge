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
def chunk_IHDR(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    if length < 13:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            return s
        msg = 'Truncated IHDR chunk'
        raise ValueError(msg)
    self.im_size = (i32(s, 0), i32(s, 4))
    try:
        self.im_mode, self.im_rawmode = _MODES[s[8], s[9]]
    except Exception:
        pass
    if s[12]:
        self.im_info['interlace'] = 1
    if s[11]:
        msg = 'unknown filter category'
        raise SyntaxError(msg)
    return s