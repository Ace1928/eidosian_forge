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
def compileLigActions(self):
    result = []
    for i, action in enumerate(self.Actions):
        last = i == len(self.Actions) - 1
        value = action.GlyphIndexDelta & 1073741823
        value |= 2147483648 if last else 0
        value |= 1073741824 if action.Store else 0
        result.append(struct.pack('>L', value))
    return bytesjoin(result)