import itertools
import logging
from typing import Callable, Iterable, Optional, Mapping
from fontTools.misc.roundTools import otRound
from fontTools.ttLib import ttFont
from fontTools.ttLib.tables import _g_l_y_f
from fontTools.ttLib.tables import _h_m_t_x
from fontTools.pens.ttGlyphPen import TTGlyphPen
import pathops
def removeTTGlyphOverlaps(glyphName: str, glyphSet: _TTGlyphMapping, glyfTable: _g_l_y_f.table__g_l_y_f, hmtxTable: _h_m_t_x.table__h_m_t_x, removeHinting: bool=True) -> bool:
    glyph = glyfTable[glyphName]
    if glyph.numberOfContours > 0 or (glyph.isComposite() and componentsOverlap(glyph, glyphSet)):
        path = skPathFromGlyph(glyphName, glyphSet)
        path2 = _simplify(path, glyphName)
        if {tuple(c) for c in path.contours} != {tuple(c) for c in path2.contours}:
            glyfTable[glyphName] = glyph = ttfGlyphFromSkPath(path2)
            assert not glyph.program
            width, lsb = hmtxTable[glyphName]
            if lsb != glyph.xMin:
                hmtxTable[glyphName] = (width, glyph.xMin)
            return True
    if removeHinting:
        glyph.removeHinting()
    return False