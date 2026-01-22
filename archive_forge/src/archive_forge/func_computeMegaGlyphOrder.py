from fontTools.merge.unicode import is_Default_Ignorable
from fontTools.pens.recordingPen import DecomposingRecordingPen
import logging
def computeMegaGlyphOrder(merger, glyphOrders):
    """Modifies passed-in glyphOrders to reflect new glyph names.
    Stores merger.glyphOrder."""
    megaOrder = {}
    for glyphOrder in glyphOrders:
        for i, glyphName in enumerate(glyphOrder):
            if glyphName in megaOrder:
                n = megaOrder[glyphName]
                while glyphName + '.' + repr(n) in megaOrder:
                    n += 1
                megaOrder[glyphName] = n
                glyphName += '.' + repr(n)
                glyphOrder[i] = glyphName
            megaOrder[glyphName] = 1
    merger.glyphOrder = megaOrder = list(megaOrder.keys())