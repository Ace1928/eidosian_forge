from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import safeEval
import array
from collections import Counter, defaultdict
import io
import logging
import struct
import sys
def calcInferredDeltas(self, origCoords, endPts):
    from fontTools.varLib.iup import iup_delta
    if self.getCoordWidth() == 1:
        raise TypeError("Only 'gvar' TupleVariation can have inferred deltas")
    if None in self.coordinates:
        if len(self.coordinates) != len(origCoords):
            raise ValueError('Expected len(origCoords) == %d; found %d' % (len(self.coordinates), len(origCoords)))
        self.coordinates = iup_delta(self.coordinates, origCoords, endPts)