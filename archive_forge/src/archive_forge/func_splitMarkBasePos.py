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
def splitMarkBasePos(oldSubTable, newSubTable, overflowRecord):
    classCount = oldSubTable.ClassCount
    if classCount < 2:
        return False
    oldClassCount = classCount // 2
    newClassCount = classCount - oldClassCount
    oldMarkCoverage, oldMarkRecords = ([], [])
    newMarkCoverage, newMarkRecords = ([], [])
    for glyphName, markRecord in zip(oldSubTable.MarkCoverage.glyphs, oldSubTable.MarkArray.MarkRecord):
        if markRecord.Class < oldClassCount:
            oldMarkCoverage.append(glyphName)
            oldMarkRecords.append(markRecord)
        else:
            markRecord.Class -= oldClassCount
            newMarkCoverage.append(glyphName)
            newMarkRecords.append(markRecord)
    oldBaseRecords, newBaseRecords = ([], [])
    for rec in oldSubTable.BaseArray.BaseRecord:
        oldBaseRecord, newBaseRecord = (rec.__class__(), rec.__class__())
        oldBaseRecord.BaseAnchor = rec.BaseAnchor[:oldClassCount]
        newBaseRecord.BaseAnchor = rec.BaseAnchor[oldClassCount:]
        oldBaseRecords.append(oldBaseRecord)
        newBaseRecords.append(newBaseRecord)
    newSubTable.Format = oldSubTable.Format
    oldSubTable.MarkCoverage.glyphs = oldMarkCoverage
    newSubTable.MarkCoverage = oldSubTable.MarkCoverage.__class__()
    newSubTable.MarkCoverage.glyphs = newMarkCoverage
    newSubTable.BaseCoverage = oldSubTable.BaseCoverage
    oldSubTable.ClassCount = oldClassCount
    newSubTable.ClassCount = newClassCount
    oldSubTable.MarkArray.MarkRecord = oldMarkRecords
    newSubTable.MarkArray = oldSubTable.MarkArray.__class__()
    newSubTable.MarkArray.MarkRecord = newMarkRecords
    oldSubTable.MarkArray.MarkCount = len(oldMarkRecords)
    newSubTable.MarkArray.MarkCount = len(newMarkRecords)
    oldSubTable.BaseArray.BaseRecord = oldBaseRecords
    newSubTable.BaseArray = oldSubTable.BaseArray.__class__()
    newSubTable.BaseArray.BaseRecord = newBaseRecords
    oldSubTable.BaseArray.BaseCount = len(oldBaseRecords)
    newSubTable.BaseArray.BaseCount = len(newBaseRecords)
    return True