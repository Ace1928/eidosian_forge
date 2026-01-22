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
def packCharset(charset, isCID, strings):
    fmt = 1
    ranges = []
    first = None
    end = 0
    if isCID:
        getNameID = getCIDfromName
    else:
        getNameID = getSIDfromName
    for name in charset[1:]:
        SID = getNameID(name, strings)
        if first is None:
            first = SID
        elif end + 1 != SID:
            nLeft = end - first
            if nLeft > 255:
                fmt = 2
            ranges.append((first, nLeft))
            first = SID
        end = SID
    if end:
        nLeft = end - first
        if nLeft > 255:
            fmt = 2
        ranges.append((first, nLeft))
    data = [packCard8(fmt)]
    if fmt == 1:
        nLeftFunc = packCard8
    else:
        nLeftFunc = packCard16
    for first, nLeft in ranges:
        data.append(packCard16(first) + nLeftFunc(nLeft))
    return bytesjoin(data)