from fontTools.misc.fixedTools import (
from fontTools.misc.textTools import bytechr, byteord, bytesjoin, strjoin
from fontTools.pens.boundsPen import BoundsPen
import struct
import logging
def op_blend(self, index):
    if self.numRegions == 0:
        self.numRegions = self.private.getNumRegions()
    numBlends = self.pop()
    numOps = numBlends * (self.numRegions + 1)
    if self.blender is None:
        del self.operandStack[-(numOps - numBlends):]
    else:
        argi = len(self.operandStack) - numOps
        end_args = tuplei = argi + numBlends
        while argi < end_args:
            next_ti = tuplei + self.numRegions
            deltas = self.operandStack[tuplei:next_ti]
            delta = self.blender(self.vsIndex, deltas)
            self.operandStack[argi] += delta
            tuplei = next_ti
            argi += 1
        self.operandStack[end_args:] = []