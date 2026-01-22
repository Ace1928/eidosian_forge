from io import BytesIO
import sys
import array
import struct
from collections import OrderedDict
from fontTools.misc import sstruct
from fontTools.misc.arrayTools import calcIntBounds
from fontTools.misc.textTools import Tag, bytechr, byteord, bytesjoin, pad
from fontTools.ttLib import (
from fontTools.ttLib.sfnt import (
from fontTools.ttLib.tables import ttProgram, _g_l_y_f
import logging
def transformTable(self, tag):
    """Return transformed table data, or None if some pre-conditions aren't
        met -- in which case, the non-transformed table data will be used.
        """
    if tag == 'loca':
        data = b''
    elif tag == 'glyf':
        for tag in ('maxp', 'head', 'loca', 'glyf'):
            self._decompileTable(tag)
        glyfTable = self.ttFont['glyf']
        data = glyfTable.transform(self.ttFont)
    elif tag == 'hmtx':
        if 'glyf' not in self.tables:
            return
        for tag in ('maxp', 'head', 'hhea', 'loca', 'glyf', 'hmtx'):
            self._decompileTable(tag)
        hmtxTable = self.ttFont['hmtx']
        data = hmtxTable.transform(self.ttFont)
    else:
        raise TTLibError("Transform for table '%s' is unknown" % tag)
    return data