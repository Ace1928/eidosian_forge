from fontTools.misc import sstruct
from . import DefaultTable
from fontTools.misc.textTools import bytesjoin, safeEval
from .BitmapGlyphMetrics import (
import struct
import itertools
from collections import deque
import logging
def padBitmapData(self, data):
    assert len(data) <= self.imageSize, 'Data in indexSubTable format %d must be less than the fixed size.' % self.indexFormat
    pad = (self.imageSize - len(data)) * b'\x00'
    return data + pad