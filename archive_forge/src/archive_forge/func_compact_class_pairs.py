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
def compact_class_pairs(font: TTFont, level: int, subtable: otTables.PairPos) -> List[otTables.PairPos]:
    from fontTools.otlLib.builder import buildPairPosClassesSubtable
    subtables = []
    classes1: DefaultDict[int, List[str]] = defaultdict(list)
    for g in subtable.Coverage.glyphs:
        classes1[subtable.ClassDef1.classDefs.get(g, 0)].append(g)
    classes2: DefaultDict[int, List[str]] = defaultdict(list)
    for g, i in subtable.ClassDef2.classDefs.items():
        classes2[i].append(g)
    all_pairs = {}
    for i, class1 in enumerate(subtable.Class1Record):
        for j, class2 in enumerate(class1.Class2Record):
            if is_really_zero(class2):
                continue
            all_pairs[tuple(sorted(classes1[i])), tuple(sorted(classes2[j]))] = (getattr(class2, 'Value1', None), getattr(class2, 'Value2', None))
    grouped_pairs = cluster_pairs_by_class2_coverage_custom_cost(font, all_pairs, level)
    for pairs in grouped_pairs:
        subtables.append(buildPairPosClassesSubtable(pairs, font.getReverseGlyphMap()))
    return subtables