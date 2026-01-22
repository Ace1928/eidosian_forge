from fontTools.misc import sstruct
from fontTools.misc.textTools import (
from .BitmapGlyphMetrics import (
from . import DefaultTable
import itertools
import os
import struct
import logging
def getRow(self, row, bitDepth=1, metrics=None, reverseBytes=False):
    if metrics is None:
        metrics = self.metrics
    assert 0 <= row and row < metrics.height, 'Illegal row access in bitmap'
    byteRange = self._getByteRange(row, bitDepth, metrics)
    data = self.imageData[slice(*byteRange)]
    if reverseBytes:
        data = _reverseBytes(data)
    return data