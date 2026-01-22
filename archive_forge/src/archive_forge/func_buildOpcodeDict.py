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
def buildOpcodeDict(table):
    d = {}
    for op, name, arg, default, conv in table:
        if isinstance(op, tuple):
            op = bytechr(op[0]) + bytechr(op[1])
        else:
            op = bytechr(op)
        d[name] = (op, arg)
    return d