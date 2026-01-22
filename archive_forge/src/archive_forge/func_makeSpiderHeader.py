from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
def makeSpiderHeader(im):
    nsam, nrow = im.size
    lenbyt = nsam * 4
    labrec = int(1024 / lenbyt)
    if 1024 % lenbyt != 0:
        labrec += 1
    labbyt = labrec * lenbyt
    nvalues = int(labbyt / 4)
    if nvalues < 23:
        return []
    hdr = [0.0] * nvalues
    hdr[1] = 1.0
    hdr[2] = float(nrow)
    hdr[3] = float(nrow)
    hdr[5] = 1.0
    hdr[12] = float(nsam)
    hdr[13] = float(labrec)
    hdr[22] = float(labbyt)
    hdr[23] = float(lenbyt)
    hdr = hdr[1:]
    hdr.append(0.0)
    return [struct.pack('f', v) for v in hdr]