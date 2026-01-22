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
def parseCursive(lines, font, _lookupMap=None):
    records = {}
    for line in lines:
        assert len(line) in [3, 4], line
        idx, klass = {'entry': (0, ot.EntryAnchor), 'exit': (1, ot.ExitAnchor)}[line[0]]
        glyph = makeGlyph(line[1])
        if glyph not in records:
            records[glyph] = [None, None]
        assert records[glyph][idx] is None, (glyph, idx)
        records[glyph][idx] = makeAnchor(line[2:], klass)
    return otl.buildCursivePosSubtable(records, font.getReverseGlyphMap())