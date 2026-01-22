from fontTools.misc import sstruct
from fontTools.misc.roundTools import otRound
from fontTools.misc.textTools import safeEval, num2binary, binary2num
from fontTools.ttLib.tables import DefaultTable
import bisect
import logging
def recalcUnicodeRanges(self, ttFont, pruneOnly=False):
    """Intersect the codepoints in the font's Unicode cmap subtables with
        the Unicode block ranges defined in the OpenType specification (v1.7),
        and set the respective 'ulUnicodeRange*' bits if there is at least ONE
        intersection.
        If 'pruneOnly' is True, only clear unused bits with NO intersection.
        """
    unicodes = set()
    for table in ttFont['cmap'].tables:
        if table.isUnicode():
            unicodes.update(table.cmap.keys())
    if pruneOnly:
        empty = intersectUnicodeRanges(unicodes, inverse=True)
        bits = self.getUnicodeRanges() - empty
    else:
        bits = intersectUnicodeRanges(unicodes)
    self.setUnicodeRanges(bits)
    return bits