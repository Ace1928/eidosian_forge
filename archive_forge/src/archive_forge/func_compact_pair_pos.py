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
def compact_pair_pos(font: TTFont, level: int, subtables: Sequence[otTables.PairPos]) -> Sequence[otTables.PairPos]:
    new_subtables = []
    for subtable in subtables:
        if subtable.Format == 1:
            new_subtables.append(subtable)
        elif subtable.Format == 2:
            new_subtables.extend(compact_class_pairs(font, level, subtable))
    return new_subtables