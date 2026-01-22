from __future__ import annotations
import os
import struct
import sys
from . import Image, ImageFile
def isSpiderHeader(t):
    h = (99,) + t
    for i in [1, 2, 5, 12, 13, 22, 23]:
        if not isInt(h[i]):
            return 0
    iform = int(h[5])
    if iform not in iforms:
        return 0
    labrec = int(h[13])
    labbyt = int(h[22])
    lenbyt = int(h[23])
    if labbyt != labrec * lenbyt:
        return 0
    return labbyt