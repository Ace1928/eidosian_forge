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
@property
def coverage_bytes(self):
    format1_bytes = 4 + sum((len(self.ctx.all_class1[i]) for i in self.indices)) * 2
    ranges = sorted(chain.from_iterable((self.ctx.all_class1_data[i][0] for i in self.indices)))
    merged_range_count = 0
    last = None
    for start, end in ranges:
        if last is not None and start != last + 1:
            merged_range_count += 1
        last = end
    format2_bytes = 4 + merged_range_count * 6
    return min(format1_bytes, format2_bytes)