from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def decompileSharedTuples(axisTags, sharedTupleCount, data, offset):
    result = []
    for _ in range(sharedTupleCount):
        t, offset = TupleVariation.decompileCoord_(axisTags, data, offset)
        result.append(t)
    return result