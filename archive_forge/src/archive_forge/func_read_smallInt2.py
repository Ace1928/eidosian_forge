from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def read_smallInt2(self, b0, data, index):
    b1 = byteord(data[index])
    return (-(b0 - 251) * 256 - b1 - 108, index + 1)