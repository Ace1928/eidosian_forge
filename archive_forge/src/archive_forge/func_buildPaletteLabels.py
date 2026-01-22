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
def buildPaletteLabels(labels: Iterable[_OptionalLocalizedString], nameTable: _n_a_m_e.table__n_a_m_e) -> List[Optional[int]]:
    return [nameTable.addMultilingualName(l, mac=False) if isinstance(l, dict) else C_P_A_L_.table_C_P_A_L_.NO_NAME_ID if l is None else nameTable.addMultilingualName({'en': l}, mac=False) for l in labels]