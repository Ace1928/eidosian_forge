from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
def readFormat4(self, reader, font):
    mapping = {}
    pos = reader.pos - 2
    unitSize = reader.readUShort()
    assert unitSize >= 6, unitSize
    for i in range(reader.readUShort()):
        reader.seek(pos + i * unitSize + 12)
        last = reader.readUShort()
        first = reader.readUShort()
        offset = reader.readUShort()
        if last != 65535:
            dataReader = reader.getSubReader(0)
            dataReader.seek(pos + offset)
            data = self.converter.readArray(dataReader, font, tableDict=None, count=last - first + 1)
            for k, v in enumerate(data):
                mapping[font.getGlyphName(first + k)] = v
    return mapping