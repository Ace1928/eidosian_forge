import collections
import copy
import enum
from functools import partial
from math import ceil, log
from typing import (
from fontTools.misc.arrayTools import intRect
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.treeTools import build_n_ary_tree
from fontTools.ttLib.tables import C_O_L_R_
from fontTools.ttLib.tables import C_P_A_L_
from fontTools.ttLib.tables import _n_a_m_e
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otTables import ExtendMode, CompositeMode
from .errors import ColorLibError
from .geometry import round_start_circle_stable_containment
from .table_builder import BuildCallback, TableBuilder
def buildClipBox(clipBox: _ClipBoxInput) -> ot.ClipBox:
    if isinstance(clipBox, ot.ClipBox):
        return clipBox
    n = len(clipBox)
    clip = ot.ClipBox()
    if n not in (4, 5):
        raise ValueError(f'Invalid ClipBox: expected 4 or 5 values, found {n}')
    clip.xMin, clip.yMin, clip.xMax, clip.yMax = intRect(clipBox[:4])
    clip.Format = int(n == 5) + 1
    if n == 5:
        clip.VarIndexBase = int(clipBox[4])
    return clip