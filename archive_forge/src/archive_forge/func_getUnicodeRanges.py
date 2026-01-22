from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def getUnicodeRanges(self):
    """Return the set of 'ulUnicodeRange*' bits currently enabled."""
    bits = set()
    ul1, ul2 = (self.ulUnicodeRange1, self.ulUnicodeRange2)
    ul3, ul4 = (self.ulUnicodeRange3, self.ulUnicodeRange4)
    for i in range(32):
        if ul1 & 1 << i:
            bits.add(i)
        if ul2 & 1 << i:
            bits.add(i + 32)
        if ul3 & 1 << i:
            bits.add(i + 64)
        if ul4 & 1 << i:
            bits.add(i + 96)
    return bits