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
def chunk_pHYs(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    if length < 9:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            return s
        msg = 'Truncated pHYs chunk'
        raise ValueError(msg)
    px, py = (i32(s, 0), i32(s, 4))
    unit = s[8]
    if unit == 1:
        dpi = (px * 0.0254, py * 0.0254)
        self.im_info['dpi'] = dpi
    elif unit == 0:
        self.im_info['aspect'] = (px, py)
    return s