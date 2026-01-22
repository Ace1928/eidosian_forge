from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def encodeDeltaRunAsZeroes_(deltas, offset, bytearr):
    pos = offset
    numDeltas = len(deltas)
    while pos < numDeltas and deltas[pos] == 0:
        pos += 1
    runLength = pos - offset
    while runLength >= 64:
        bytearr.append(DELTAS_ARE_ZERO | 63)
        runLength -= 64
    if runLength:
        bytearr.append(DELTAS_ARE_ZERO | runLength - 1)
    return pos