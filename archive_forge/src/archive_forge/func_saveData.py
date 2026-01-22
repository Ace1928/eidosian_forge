from io import BytesIO
from types import SimpleNamespace
from fontTools.misc.textTools import Tag
from fontTools.misc import sstruct
from fontTools.ttLib import TTLibError, TTLibFileIsCollectionError
import struct
from collections import OrderedDict
import logging
def saveData(self, file, data):
    if hasattr(self.__class__, 'encodeData'):
        data = self.encodeData(data)
    self.length = len(data)
    file.seek(self.offset)
    file.write(data)