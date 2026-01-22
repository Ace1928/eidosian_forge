from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def countHints(self):
    args = self.popallWidth()
    self.hintCount = self.hintCount + len(args) // 2