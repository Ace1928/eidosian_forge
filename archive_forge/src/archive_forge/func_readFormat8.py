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
def readFormat8(self, reader, font):
    first = reader.readUShort()
    count = reader.readUShort()
    data = self.converter.readArray(reader, font, tableDict=None, count=count)
    return {font.getGlyphName(first + k): value for k, value in enumerate(data)}