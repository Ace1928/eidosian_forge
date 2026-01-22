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
def processAxis(font, planFunc, axisTag, axisName, values, samples=None, glyphs=None, designLimits=None, pins=None, sanitize=False, plot=False):
    """Process a single axis."""
    axisLimits = None
    for axis in font['fvar'].axes:
        if axis.axisTag == axisTag:
            axisLimits = axis
            break
    if axisLimits is None:
        return ''
    axisLimits = (axisLimits.minValue, axisLimits.defaultValue, axisLimits.maxValue)
    log.info('Planning %s axis.', axisName)
    if 'avar' in font:
        existingMapping = font['avar'].segments[axisTag]
        font['avar'].segments[axisTag] = {}
    else:
        existingMapping = None
    if values is not None and isinstance(values, str):
        values = [float(w) for w in values.split()]
    if designLimits is not None and isinstance(designLimits, str):
        designLimits = [float(d) for d in options.designLimits.split(':')]
        assert len(designLimits) == 3 and designLimits[0] <= designLimits[1] <= designLimits[2]
    else:
        designLimits = None
    if pins is not None and isinstance(pins, str):
        newPins = {}
        for pin in pins.split():
            before, after = pin.split(':')
            newPins[float(before)] = float(after)
        pins = newPins
        del newPins
    mapping, mappingNormalized = planFunc(font.getGlyphSet, axisLimits, values, samples=samples, glyphs=glyphs, designLimits=designLimits, pins=pins, sanitize=sanitize)
    if plot:
        from matplotlib import pyplot
        pyplot.plot(sorted(mappingNormalized), [mappingNormalized[k] for k in sorted(mappingNormalized)])
        pyplot.show()
    if existingMapping is not None:
        log.info('Existing %s mapping:\n%s', axisName, pformat(existingMapping))
    if mapping:
        if 'avar' not in font:
            addEmptyAvar(font)
        font['avar'].segments[axisTag] = mappingNormalized
    elif 'avar' in font:
        font['avar'].segments[axisTag] = {}
    designspaceSnippet = makeDesignspaceSnippet(axisTag, axisName, axisLimits, mapping)
    return designspaceSnippet