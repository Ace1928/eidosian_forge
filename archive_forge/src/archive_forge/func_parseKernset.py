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
def parseKernset(lines, font, _lookupMap=None):
    typ = lines.peeks()[0].split()[0].lower()
    if typ in ('left', 'right'):
        with lines.until(('firstclass definition begin', 'secondclass definition begin')):
            return parsePair(lines, font)
    return parsePair(lines, font)