from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
def removeSkipGlyphs(self):

    def isValidLocation(args):
        name, (startByte, endByte) = args
        return startByte < endByte
    dataPairs = list(filter(isValidLocation, zip(self.names, self.locations)))
    self.names, self.locations = list(map(list, zip(*dataPairs)))