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
def chunk_iCCP(self, pos, length):
    s = ImageFile._safe_read(self.fp, length)
    i = s.find(b'\x00')
    logger.debug('iCCP profile name %r', s[:i])
    logger.debug('Compression method %s', s[i])
    comp_method = s[i]
    if comp_method != 0:
        msg = f'Unknown compression method {comp_method} in iCCP chunk'
        raise SyntaxError(msg)
    try:
        icc_profile = _safe_zlib_decompress(s[i + 2:])
    except ValueError:
        if ImageFile.LOAD_TRUNCATED_IMAGES:
            icc_profile = None
        else:
            raise
    except zlib.error:
        icc_profile = None
    self.im_info['icc_profile'] = icc_profile
    return s