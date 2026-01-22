from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_vvcurveto(self, index):
    """dx1? {dya dxb dyb dyc}+ vvcurveto"""
    args = self.popall()
    if len(args) % 2:
        dx1 = args[0]
        args = args[1:]
    else:
        dx1 = 0
    for i in range(0, len(args), 4):
        dya, dxb, dyb, dyc = args[i:i + 4]
        self.rCurveTo((dx1, dya), (dxb, dyb), (0, dyc))
        dx1 = 0