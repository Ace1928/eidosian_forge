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
def packEncoding0(charset, encoding, strings):
    fmt = 0
    m = {}
    for code in range(len(encoding)):
        name = encoding[code]
        if name != '.notdef':
            m[name] = code
    codes = []
    for name in charset[1:]:
        code = m.get(name)
        codes.append(code)
    while codes and codes[-1] is None:
        codes.pop()
    data = [packCard8(fmt), packCard8(len(codes))]
    for code in codes:
        if code is None:
            code = 0
        data.append(packCard8(code))
    return bytesjoin(data)