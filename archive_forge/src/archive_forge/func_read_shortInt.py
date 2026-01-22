from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def read_shortInt(self, b0, data, index):
    value, = struct.unpack('>h', data[index:index + 2])
    return (value, index + 2)