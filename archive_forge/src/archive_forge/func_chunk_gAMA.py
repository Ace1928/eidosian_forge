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
def chunk_gAMA(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    self.im_info['gamma'] = i32(s) / 100000.0
    return s