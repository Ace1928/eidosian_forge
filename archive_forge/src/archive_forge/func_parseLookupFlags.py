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
def parseLookupFlags(lines):
    flags = 0
    filterset = None
    allFlags = ['righttoleft', 'ignorebaseglyphs', 'ignoreligatures', 'ignoremarks', 'markattachmenttype', 'markfiltertype']
    while lines.peeks()[0].lower() in allFlags:
        line = next(lines)
        flag = {'righttoleft': 1, 'ignorebaseglyphs': 2, 'ignoreligatures': 4, 'ignoremarks': 8}.get(line[0].lower())
        if flag:
            assert line[1].lower() in ['yes', 'no'], line[1]
            if line[1].lower() == 'yes':
                flags |= flag
            continue
        if line[0].lower() == 'markattachmenttype':
            flags |= int(line[1]) << 8
            continue
        if line[0].lower() == 'markfiltertype':
            flags |= 16
            filterset = int(line[1])
    return (flags, filterset)