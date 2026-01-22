import itertools
import logging
from typing import Callable, Iterable, Optional, Mapping
from fontTools.misc.roundTools import otRound
from fontTools.ttLib import ttFont
from fontTools.ttLib.tables import _g_l_y_f
from fontTools.ttLib.tables import _h_m_t_x
from fontTools.pens.ttGlyphPen import TTGlyphPen
import pathops
def skPathFromGlyph(glyphName: str, glyphSet: _TTGlyphMapping) -> pathops.Path:
    path = pathops.Path()
    pathPen = path.getPen(glyphSet=glyphSet)
    glyphSet[glyphName].draw(pathPen)
    return path