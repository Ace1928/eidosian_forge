import logging
import os
from collections import defaultdict, namedtuple
from functools import reduce
from itertools import chain
from math import log2
from typing import DefaultDict, Dict, Iterable, List, Sequence, Tuple
from fontTools.config import OPTIONS
from fontTools.misc.intTools import bit_count, bit_indices
from fontTools.ttLib import TTFont
from fontTools.ttLib.tables import otBase, otTables
def compact_ext_lookup(font: TTFont, level: int, lookup: otTables.Lookup) -> None:
    new_subtables = compact_pair_pos(font, level, [ext_subtable.ExtSubTable for ext_subtable in lookup.SubTable])
    new_ext_subtables = []
    for subtable in new_subtables:
        ext_subtable = otTables.ExtensionPos()
        ext_subtable.Format = 1
        ext_subtable.ExtSubTable = subtable
        new_ext_subtables.append(ext_subtable)
    lookup.SubTable = new_ext_subtables
    lookup.SubTableCount = len(new_ext_subtables)