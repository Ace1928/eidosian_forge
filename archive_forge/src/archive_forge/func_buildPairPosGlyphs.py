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
def buildPairPosGlyphs(pairs, glyphMap):
    """Builds a list of glyph-based pair adjustment (GPOS2 format 1) subtables.

    This organises a list of pair positioning adjustments into subtables based
    on common value record formats.

    Note that if you are implementing a layout compiler, you may find it more
    flexible to use
    :py:class:`fontTools.otlLib.lookupBuilders.PairPosBuilder`
    instead.

    Example::

        pairs = {
            ("K", "W"): ( buildValue(xAdvance=+5), buildValue() ),
            ("K", "V"): ( buildValue(xAdvance=+5), buildValue() ),
            # ...
        }

        subtables = buildPairPosGlyphs(pairs, font.getReverseGlyphMap())

    Args:
        pairs (dict): Pair positioning data; the keys being a two-element
            tuple of glyphnames, and the values being a two-element
            tuple of ``otTables.ValueRecord`` objects.
        glyphMap: a glyph name to ID map, typically returned from
            ``font.getReverseGlyphMap()``.

    Returns:
        A list of ``otTables.PairPos`` objects.
    """
    p = {}
    for (glyphA, glyphB), (valA, valB) in pairs.items():
        formatA = valA.getFormat() if valA is not None else 0
        formatB = valB.getFormat() if valB is not None else 0
        pos = p.setdefault((formatA, formatB), {})
        pos[glyphA, glyphB] = (valA, valB)
    return [buildPairPosGlyphsSubtable(pos, glyphMap, formatA, formatB) for (formatA, formatB), pos in sorted(p.items())]