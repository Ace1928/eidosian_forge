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
def buildFormat2(self, writer, font, values):
    segStart, segValue = values[0]
    segEnd = segStart
    segments = []
    for glyphID, curValue in values[1:]:
        if glyphID != segEnd + 1 or curValue != segValue:
            segments.append((segStart, segEnd, segValue))
            segStart = segEnd = glyphID
            segValue = curValue
        else:
            segEnd = glyphID
    segments.append((segStart, segEnd, segValue))
    valueSize = self.converter.staticSize
    numUnits, unitSize = (len(segments) + 1, valueSize + 4)
    return (2 + self.BIN_SEARCH_HEADER_SIZE + numUnits * unitSize, 2, lambda: self.writeFormat2(writer, font, segments))