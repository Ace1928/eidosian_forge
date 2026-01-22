from fontTools.misc.textTools import bytesjoin, safeEval
from . import DefaultTable
import array
from collections import namedtuple
import struct
import sys
@classmethod
def fromHex(cls, value):
    if value[0] == '#':
        value = value[1:]
    red = int(value[0:2], 16)
    green = int(value[2:4], 16)
    blue = int(value[4:6], 16)
    alpha = int(value[6:8], 16) if len(value) >= 8 else 255
    return cls(red=red, green=green, blue=blue, alpha=alpha)