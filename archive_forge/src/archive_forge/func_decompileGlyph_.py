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
def decompileGlyph_(pointCount, sharedTuples, axisTags, data):
    if len(data) < 4:
        return []
    tupleVariationCount, offsetToData = struct.unpack('>HH', data[:4])
    dataPos = offsetToData
    return tv.decompileTupleVariationStore('gvar', axisTags, tupleVariationCount, pointCount, sharedTuples, data, 4, offsetToData)