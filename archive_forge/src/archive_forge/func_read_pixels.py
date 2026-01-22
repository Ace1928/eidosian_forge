import os
import zlib
import time  # noqa
import logging
import numpy as np
def read_pixels(bb, i, tagType, L1):
    """With pf's seed after the recordheader, reads the pixeldata."""
    charId = bb[i:i + 2]
    i += 2
    format = ord(bb[i:i + 1])
    i += 1
    width = bits2int(bb[i:i + 2], 16)
    i += 2
    height = bits2int(bb[i:i + 2], 16)
    i += 2
    if format != 5:
        logger.warning('Can only read 24bit or 32bit RGB(A) lossless images.')
    else:
        offset = 2 + 1 + 2 + 2
        bb2 = bb[i:i + (L1 - offset)]
        data = zlib.decompress(bb2)
        a = np.frombuffer(data, dtype=np.uint8)
        if tagType == 20:
            try:
                a.shape = (height, width, 3)
            except Exception:
                logger.warning('Cannot read image due to byte alignment')
        if tagType == 36:
            a.shape = (height, width, 4)
            b = a
            a = np.zeros_like(a)
            a[:, :, 0] = b[:, :, 1]
            a[:, :, 1] = b[:, :, 2]
            a[:, :, 2] = b[:, :, 3]
            a[:, :, 3] = b[:, :, 0]
        return a