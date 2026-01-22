from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_callothersubr(self, index):
    subrIndex = self.pop()
    nArgs = self.pop()
    if subrIndex == 0 and nArgs == 3:
        self.doFlex()
        self.flexing = 0
    elif subrIndex == 1 and nArgs == 0:
        self.flexing = 1