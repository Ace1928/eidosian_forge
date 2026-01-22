from collections import namedtuple, OrderedDict
import os
from fontTools.misc.fixedTools import fixedToFloat
from fontTools.misc.roundTools import otRound
from fontTools import ttLib
from fontTools.ttLib.tables import otTables as ot
from fontTools.ttLib.tables.otBase import (
from fontTools.ttLib.tables import otBase
from fontTools.feaLib.ast import STATNameStatement
from fontTools.otlLib.optimize.gpos import (
from fontTools.otlLib.error import OpenTypeLibError
from functools import reduce
import logging
import copy
def buildLigCaretList(coords, points, glyphMap):
    """Builds a ligature caret list table.

    Ligatures appear as a single glyph representing multiple characters; however
    when, for example, editing text containing a ``f_i`` ligature, the user may
    want to place the cursor between the ``f`` and the ``i``. The ligature caret
    list in the GDEF table specifies the position to display the "caret" (the
    character insertion indicator, typically a flashing vertical bar) "inside"
    the ligature to represent an insertion point. The insertion positions may
    be specified either by coordinate or by contour point.

    Example::

        coords = {
            "f_f_i": [300, 600] # f|fi cursor at 300 units, ff|i cursor at 600.
        }
        points = {
            "c_t": [28] # c|t cursor appears at coordinate of contour point 28.
        }
        ligcaretlist = buildLigCaretList(coords, points, font.getReverseGlyphMap())

    Args:
        coords: A mapping between glyph names and a list of coordinates for
            the insertion point of each ligature component after the first one.
        points: A mapping between glyph names and a list of contour points for
            the insertion point of each ligature component after the first one.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A ``otTables.LigCaretList`` object if any carets are present, or
            ``None`` otherwise."""
    glyphs = set(coords.keys()) if coords else set()
    if points:
        glyphs.update(points.keys())
    carets = {g: buildLigGlyph(coords.get(g), points.get(g)) for g in glyphs}
    carets = {g: c for g, c in carets.items() if c is not None}
    if not carets:
        return None
    self = ot.LigCaretList()
    self.Coverage = buildCoverage(carets.keys(), glyphMap)
    self.LigGlyph = [carets[g] for g in self.Coverage.glyphs]
    self.LigGlyphCount = len(self.LigGlyph)
    return self