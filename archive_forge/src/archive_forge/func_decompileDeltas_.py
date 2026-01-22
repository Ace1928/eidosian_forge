from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def decompileDeltas_(numDeltas, data, offset):
    """(numDeltas, data, offset) --> ([delta, delta, ...], newOffset)"""
    result = []
    pos = offset
    while len(result) < numDeltas:
        runHeader = data[pos]
        pos += 1
        numDeltasInRun = (runHeader & DELTA_RUN_COUNT_MASK) + 1
        if runHeader & DELTAS_ARE_ZERO != 0:
            result.extend([0] * numDeltasInRun)
        else:
            if runHeader & DELTAS_ARE_WORDS != 0:
                deltas = array.array('h')
                deltasSize = numDeltasInRun * 2
            else:
                deltas = array.array('b')
                deltasSize = numDeltasInRun
            deltas.frombytes(data[pos:pos + deltasSize])
            if sys.byteorder != 'big':
                deltas.byteswap()
            assert len(deltas) == numDeltasInRun
            pos += deltasSize
            result.extend(deltas)
    assert len(result) == numDeltas
    return (result, pos)