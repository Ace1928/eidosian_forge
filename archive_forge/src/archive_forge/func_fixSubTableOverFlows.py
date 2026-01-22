import copy
from enum import IntEnum
from functools import reduce
from math import radians
import itertools
from collections import defaultdict, namedtuple
from fontTools.ttLib.tables.otTraverse import dfs_base_table
from fontTools.misc.arrayTools import quantizeRect
from fontTools.misc.roundTools import otRound
from fontTools.misc.transform import Transform, Identity
from fontTools.misc.textTools import bytesjoin, pad, safeEval
from fontTools.pens.boundsPen import ControlBoundsPen
from fontTools.pens.transformPen import TransformPen
from .otBase import (
from fontTools.feaLib.lookupDebugInfo import LookupDebugInfo, LOOKUP_DEBUG_INFO_KEY
import logging
import struct
from typing import TYPE_CHECKING, Iterator, List, Optional, Set
def fixSubTableOverFlows(ttf, overflowRecord):
    """
    An offset has overflowed within a sub-table. We need to divide this subtable into smaller parts.
    """
    table = ttf[overflowRecord.tableType].table
    lookup = table.LookupList.Lookup[overflowRecord.LookupListIndex]
    subIndex = overflowRecord.SubTableIndex
    subtable = lookup.SubTable[subIndex]
    if not hasattr(subtable, 'DontShare'):
        subtable.DontShare = True
        return True
    if hasattr(subtable, 'ExtSubTable'):
        subTableType = subtable.ExtSubTable.__class__.LookupType
        extSubTable = subtable
        subtable = extSubTable.ExtSubTable
        newExtSubTableClass = lookupTypes[overflowRecord.tableType][extSubTable.__class__.LookupType]
        newExtSubTable = newExtSubTableClass()
        newExtSubTable.Format = extSubTable.Format
        toInsert = newExtSubTable
        newSubTableClass = lookupTypes[overflowRecord.tableType][subTableType]
        newSubTable = newSubTableClass()
        newExtSubTable.ExtSubTable = newSubTable
    else:
        subTableType = subtable.__class__.LookupType
        newSubTableClass = lookupTypes[overflowRecord.tableType][subTableType]
        newSubTable = newSubTableClass()
        toInsert = newSubTable
    if hasattr(lookup, 'SubTableCount'):
        lookup.SubTableCount = lookup.SubTableCount + 1
    try:
        splitFunc = splitTable[overflowRecord.tableType][subTableType]
    except KeyError:
        log.error("Don't know how to split %s lookup type %s", overflowRecord.tableType, subTableType)
        return False
    ok = splitFunc(subtable, newSubTable, overflowRecord)
    if ok:
        lookup.SubTable.insert(subIndex + 1, toInsert)
    return ok