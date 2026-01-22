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
def calcOffSize(largestOffset):
    if largestOffset < 256:
        offSize = 1
    elif largestOffset < 65536:
        offSize = 2
    elif largestOffset < 16777216:
        offSize = 3
    else:
        offSize = 4
    return offSize