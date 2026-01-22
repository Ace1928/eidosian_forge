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
def planWidthAxis(glyphSetFunc, axisLimits, widths=None, samples=None, glyphs=None, designLimits=None, pins=None, sanitize=False):
    """Plan a width (`wdth`) axis.

    widths: A list of width values (percentages) to plan for. If None, the default
    values are used.

    This function simply calls planAxis with values=widths, and the appropriate
    arguments. See documenation for planAxis for more information.
    """
    if widths is None:
        widths = WIDTHS
    return planAxis(measureWidth, normalizeLinear, interpolateLinear, glyphSetFunc, 'wdth', axisLimits, values=widths, samples=samples, glyphs=glyphs, designLimits=designLimits, pins=pins, sanitizeFunc=sanitizeWidth if sanitize else None)