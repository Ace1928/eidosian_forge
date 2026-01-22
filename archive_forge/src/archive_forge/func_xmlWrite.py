from fontTools.misc.fixedTools import (
from fontTools.misc.roundTools import nearestMultipleShortestRepr, otRound
from fontTools.misc.textTools import bytesjoin, tobytes, tostr, pad, safeEval
from fontTools.ttLib import getSearchRange
from .otBase import (
from .otTables import (
from itertools import zip_longest
from functools import partial
import re
import struct
from typing import Optional
import logging
def xmlWrite(self, xmlWriter, font, value, name, attrs):
    xmlWriter.simpletag(name, attrs + [('value', value)])
    flags = []
    if value & 1:
        flags.append('rightToLeft')
    if value & 2:
        flags.append('ignoreBaseGlyphs')
    if value & 4:
        flags.append('ignoreLigatures')
    if value & 8:
        flags.append('ignoreMarks')
    if value & 16:
        flags.append('useMarkFilteringSet')
    if value & 65280:
        flags.append('markAttachmentType[%i]' % (value >> 8))
    if flags:
        xmlWriter.comment(' '.join(flags))
    xmlWriter.newline()