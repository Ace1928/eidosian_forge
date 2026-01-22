from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
def encodeData(self, data):
    self.origLength = len(data)
    if not self.uncompressed:
        compressedData = compress(data, self.zlibCompressionLevel)
    if self.uncompressed or len(compressedData) >= self.origLength:
        rawData = data
        self.length = self.origLength
    else:
        rawData = compressedData
        self.length = len(rawData)
    return rawData