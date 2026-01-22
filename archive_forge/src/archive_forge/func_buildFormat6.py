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
def buildFormat6(self, writer, font, values):
    valueSize = self.converter.staticSize
    numUnits, unitSize = (len(values), valueSize + 2)
    return (2 + self.BIN_SEARCH_HEADER_SIZE + (numUnits + 1) * unitSize, 6, lambda: self.writeFormat6(writer, font, values))