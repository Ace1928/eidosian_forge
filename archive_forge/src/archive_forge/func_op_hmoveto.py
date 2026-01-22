from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_hmoveto(self, index):
    if self.flexing:
        self.push(0)
        return
    self.endPath()
    self.rMoveTo((self.popall()[0], 0))