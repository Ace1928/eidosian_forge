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
def makeMarkFilteringSets(sets, font):
    self = ot.MarkGlyphSetsDef()
    self.MarkSetTableFormat = 1
    self.MarkSetCount = 1 + max(sets.keys())
    self.Coverage = [None] * self.MarkSetCount
    for k, v in sorted(sets.items()):
        self.Coverage[k] = makeCoverage(set(v), font)
    return self