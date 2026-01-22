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
def readFormat0(self, reader, font):
    numGlyphs = len(font.getGlyphOrder())
    data = self.converter.readArray(reader, font, tableDict=None, count=numGlyphs)
    return {font.getGlyphName(k): value for k, value in enumerate(data)}