from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def compileSharedTuples(axisTags, variations, MAX_NUM_SHARED_COORDS=TUPLE_INDEX_MASK + 1):
    coordCount = Counter()
    for var in variations:
        coord = var.compileCoord(axisTags)
        coordCount[coord] += 1
    sharedCoords = sorted(coordCount.most_common(MAX_NUM_SHARED_COORDS), key=lambda item: (-item[1], item[0]))
    return [c[0] for c in sharedCoords if c[1] > 1]