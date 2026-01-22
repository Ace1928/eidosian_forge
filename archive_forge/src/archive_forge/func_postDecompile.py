from fontTools.misc import sstruct
from fontTools.misc import psCharStrings
from fontTools.misc.arrayTools import unionRect, intRect
from fontTools.misc.textTools import (
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables.otBase import OTTableWriter
from fontTools.ttLib.tables.otBase import OTTableReader
from fontTools.ttLib.tables import otTables as ot
from io import BytesIO
import struct
import logging
import re
def postDecompile(self):
    offset = self.rawDict.get('CharStrings')
    if offset is None:
        return
    self.file.seek(offset)
    if self._isCFF2:
        self.numGlyphs = readCard32(self.file)
    else:
        self.numGlyphs = readCard16(self.file)