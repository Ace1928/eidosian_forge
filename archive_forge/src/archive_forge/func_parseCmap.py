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
def parseCmap(lines, font):
    container = ttLib.getTableClass('cmap')()
    log.debug('Parsing cmap')
    tables = []
    while lines.peek() is not None:
        lines.expect('cmap subtable %d' % len(tables))
        platId, encId, fmt, lang = [parseCmapId(lines, field) for field in ('platformID', 'encodingID', 'format', 'language')]
        table = cmap_classes[fmt](fmt)
        table.platformID = platId
        table.platEncID = encId
        table.language = lang
        table.cmap = {}
        line = next(lines)
        while line[0] != 'end subtable':
            table.cmap[int(line[0], 16)] = line[1]
            line = next(lines)
        tables.append(table)
    container.tableVersion = 0
    container.tables = tables
    return container