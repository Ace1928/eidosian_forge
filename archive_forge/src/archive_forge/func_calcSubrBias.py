from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def calcSubrBias(subrs):
    nSubrs = len(subrs)
    if nSubrs < 1240:
        bias = 107
    elif nSubrs < 33900:
        bias = 1131
    else:
        bias = 32768
    return bias