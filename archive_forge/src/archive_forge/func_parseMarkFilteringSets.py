from fontTools import ttLib
from fontTools.ttLib.tables._c_m_a_p import cmap_classes
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import ValueRecord, valueRecordFormatDict
from fontTools.otlLib import builder as otl
from contextlib import contextmanager
from fontTools.ttLib import newTable
from fontTools.feaLib.lookupDebugInfo import LOOKUP_DEBUG_ENV_VAR, LOOKUP_DEBUG_INFO_KEY
from operator import setitem
import os
import logging
def parseMarkFilteringSets(lines, font):
    sets = {}
    with lines.between('set definition'):
        for line in lines:
            assert len(line) == 2, line
            glyph = makeGlyph(line[0])
            st = int(line[1])
            if st not in sets:
                sets[st] = []
            sets[st].append(glyph)
    return makeMarkFilteringSets(sets, font)