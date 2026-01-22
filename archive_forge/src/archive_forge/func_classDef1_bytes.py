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
def classDef1_bytes(self):
    biggest_index = max(self.indices, key=lambda i: len(self.ctx.all_class1[i]))
    return _classDef_bytes(self.ctx.all_class1_data, [i for i in self.indices if i != biggest_index])