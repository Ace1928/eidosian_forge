from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def compileDeltas(self):
    deltaX = []
    deltaY = []
    if self.getCoordWidth() == 2:
        for c in self.coordinates:
            if c is None:
                continue
            deltaX.append(c[0])
            deltaY.append(c[1])
    else:
        for c in self.coordinates:
            if c is None:
                continue
            deltaX.append(c)
    bytearr = bytearray()
    self.compileDeltaValues_(deltaX, bytearr)
    self.compileDeltaValues_(deltaY, bytearr)
    return bytearr