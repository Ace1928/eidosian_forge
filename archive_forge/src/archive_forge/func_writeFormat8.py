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
def writeFormat8(self, writer, font, values):
    firstGlyphID = values[0][0]
    writer.writeUShort(8)
    writer.writeUShort(firstGlyphID)
    writer.writeUShort(len(values))
    for _, value in values:
        self.converter.write(writer, font, tableDict=None, value=value, repeatIndex=None)