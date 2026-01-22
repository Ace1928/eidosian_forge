from fontTools.ttLib import newTable
from fontTools.ttLib.tables._f_v_a_r import Axis as fvarAxis
from fontTools.pens.areaPen import AreaPen
from fontTools.pens.basePen import NullPen
from fontTools.pens.statisticsPen import StatisticsPen
from fontTools.varLib.models import piecewiseLinearMap, normalizeValue
from fontTools.misc.cliTools import makeOutputFileName
import math
import logging
from pprint import pformat
def measureSlant(glyphset, glyphs=None):
    """Measure the perceptual average slant angle of the given glyphs."""
    if isinstance(glyphs, dict):
        frequencies = glyphs
    else:
        frequencies = {g: 1 for g in glyphs}
    slnt_sum = 0
    freq_sum = 0
    for glyph_name in glyphs:
        if frequencies is not None:
            frequency = frequencies.get(glyph_name, 0)
            if frequency == 0:
                continue
        else:
            frequency = 1
        glyph = glyphset[glyph_name]
        pen = StatisticsPen(glyphset=glyphset)
        glyph.draw(pen)
        mult = glyph.width * frequency
        slnt_sum += mult * pen.slant
        freq_sum += mult
    return -math.degrees(math.atan(slnt_sum / freq_sum))