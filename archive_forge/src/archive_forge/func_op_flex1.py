from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_flex1(self, index):
    dx1, dy1, dx2, dy2, dx3, dy3, dx4, dy4, dx5, dy5, d6 = self.popall()
    dx = dx1 + dx2 + dx3 + dx4 + dx5
    dy = dy1 + dy2 + dy3 + dy4 + dy5
    if abs(dx) > abs(dy):
        dx6 = d6
        dy6 = -dy
    else:
        dx6 = -dx
        dy6 = d6
    self.rCurveTo((dx1, dy1), (dx2, dy2), (dx3, dy3))
    self.rCurveTo((dx4, dy4), (dx5, dy5), (dx6, dy6))