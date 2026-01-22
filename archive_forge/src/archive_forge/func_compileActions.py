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
@staticmethod
def compileActions(font, states):
    actions, actionIndex, result = (set(), {}, b'')
    for state in states:
        for _glyphClass, trans in state.Transitions.items():
            if trans.CurrentInsertionAction is not None:
                actions.add(tuple(trans.CurrentInsertionAction))
            if trans.MarkedInsertionAction is not None:
                actions.add(tuple(trans.MarkedInsertionAction))
    for action in sorted(actions, key=lambda x: (-len(x), x)):
        if action in actionIndex:
            continue
        for start in range(0, len(action)):
            startIndex = len(result) // 2 + start
            for limit in range(start, len(action)):
                glyphs = action[start:limit + 1]
                actionIndex.setdefault(glyphs, startIndex)
        for glyph in action:
            glyphID = font.getGlyphID(glyph)
            result += struct.pack('>H', glyphID)
    return (result, actionIndex)