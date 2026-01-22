from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
def decodeData(self, rawData):
    import zlib
    if self.length == self.origLength:
        data = rawData
    else:
        assert self.length < self.origLength
        data = zlib.decompress(rawData)
        assert len(data) == self.origLength
    return data