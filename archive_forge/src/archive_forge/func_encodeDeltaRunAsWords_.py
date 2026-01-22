from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
@staticmethod
def encodeDeltaRunAsWords_(deltas, offset, bytearr):
    pos = offset
    numDeltas = len(deltas)
    while pos < numDeltas:
        value = deltas[pos]
        if value == 0:
            break
        if -128 <= value <= 127 and pos + 1 < numDeltas and (-128 <= deltas[pos + 1] <= 127):
            break
        pos += 1
    runLength = pos - offset
    while runLength >= 64:
        bytearr.append(DELTAS_ARE_WORDS | 63)
        a = array.array('h', deltas[offset:offset + 64])
        if sys.byteorder != 'big':
            a.byteswap()
        bytearr.extend(a)
        offset += 64
        runLength -= 64
    if runLength:
        bytearr.append(DELTAS_ARE_WORDS | runLength - 1)
        a = array.array('h', deltas[offset:pos])
        if sys.byteorder != 'big':
            a.byteswap()
        bytearr.extend(a)
    return pos