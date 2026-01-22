from collections import UserDict, deque
from functools import partial
from fontTools.misc import sstruct
from fontTools.misc.textTools import safeEval
from . import DefaultTable
import array
import itertools
import logging
import struct
import sys
import fontTools.ttLib.tables.TupleVariation as tv
def compileGlyph_(variations, pointCount, axisTags, sharedCoordIndices):
    tupleVariationCount, tuples, data = tv.compileTupleVariationStore(variations, pointCount, axisTags, sharedCoordIndices)
    if tupleVariationCount == 0:
        return b''
    result = [struct.pack('>HH', tupleVariationCount, 4 + len(tuples)), tuples, data]
    if (len(tuples) + len(data)) % 2 != 0:
        result.append(b'\x00')
    return b''.join(result)