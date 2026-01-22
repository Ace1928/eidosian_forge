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
def getStdCharSet(charset):
    predefinedCharSetVal = None
    predefinedCharSets = [(cffISOAdobeStringCount, cffISOAdobeStrings, 0), (cffExpertStringCount, cffIExpertStrings, 1), (cffExpertSubsetStringCount, cffExpertSubsetStrings, 2)]
    lcs = len(charset)
    for cnt, pcs, csv in predefinedCharSets:
        if predefinedCharSetVal is not None:
            break
        if lcs > cnt:
            continue
        predefinedCharSetVal = csv
        for i in range(lcs):
            if charset[i] != pcs[i]:
                predefinedCharSetVal = None
                break
    return predefinedCharSetVal