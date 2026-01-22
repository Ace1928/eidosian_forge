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
def planAxis(measureFunc, normalizeFunc, interpolateFunc, glyphSetFunc, axisTag, axisLimits, values, samples=None, glyphs=None, designLimits=None, pins=None, sanitizeFunc=None):
    """Plan an axis.

    measureFunc: callable that takes a glyphset and an optional
    list of glyphnames, and returns the glyphset-wide measurement
    to be used for the axis.

    normalizeFunc: callable that takes a measurement and a minimum
    and maximum, and normalizes the measurement into the range 0..1,
    possibly extrapolating too.

    interpolateFunc: callable that takes a normalized t value, and a
    minimum and maximum, and returns the interpolated value,
    possibly extrapolating too.

    glyphSetFunc: callable that takes a variations "location" dictionary,
    and returns a glyphset.

    axisTag: the axis tag string.

    axisLimits: a triple of minimum, default, and maximum values for
    the axis. Or an `fvar` Axis object.

    values: a list of output values to map for this axis.

    samples: the number of samples to use when sampling. Default 8.

    glyphs: a list of glyph names to use when sampling. Defaults to None,
    which will process all glyphs.

    designLimits: an optional triple of minimum, default, and maximum values
    represenging the "design" limits for the axis. If not provided, the
    axisLimits will be used.

    pins: an optional dictionary of before/after mapping entries to pin in
    the output.

    sanitizeFunc: an optional callable to call to sanitize the axis limits.
    """
    if isinstance(axisLimits, fvarAxis):
        axisLimits = (axisLimits.minValue, axisLimits.defaultValue, axisLimits.maxValue)
    minValue, defaultValue, maxValue = axisLimits
    if samples is None:
        samples = SAMPLES
    if glyphs is None:
        glyphs = glyphSetFunc({}).keys()
    if pins is None:
        pins = {}
    else:
        pins = pins.copy()
    log.info('Axis limits min %g / default %g / max %g', minValue, defaultValue, maxValue)
    triple = (minValue, defaultValue, maxValue)
    if designLimits is not None:
        log.info('Axis design-limits min %g / default %g / max %g', *designLimits)
    else:
        designLimits = triple
    if pins:
        log.info('Pins %s', sorted(pins.items()))
    pins.update({minValue: designLimits[0], defaultValue: designLimits[1], maxValue: designLimits[2]})
    out = {}
    outNormalized = {}
    axisMeasurements = {}
    for value in sorted({minValue, defaultValue, maxValue} | set(pins.keys())):
        glyphset = glyphSetFunc(location={axisTag: value})
        designValue = pins[value]
        axisMeasurements[designValue] = measureFunc(glyphset, glyphs)
    if sanitizeFunc is not None:
        log.info('Sanitizing axis limit values for the `%s` axis.', axisTag)
        sanitizeFunc(triple, designLimits, pins, axisMeasurements)
    log.debug('Calculated average value:\n%s', pformat(axisMeasurements))
    for (rangeMin, targetMin), (rangeMax, targetMax) in zip(list(sorted(pins.items()))[:-1], list(sorted(pins.items()))[1:]):
        targetValues = {w for w in values if rangeMin < w < rangeMax}
        if not targetValues:
            continue
        normalizedMin = normalizeValue(rangeMin, triple)
        normalizedMax = normalizeValue(rangeMax, triple)
        normalizedTargetMin = normalizeValue(targetMin, designLimits)
        normalizedTargetMax = normalizeValue(targetMax, designLimits)
        log.info('Planning target values %s.', sorted(targetValues))
        log.info('Sampling %u points in range %g,%g.', samples, rangeMin, rangeMax)
        valueMeasurements = axisMeasurements.copy()
        for sample in range(1, samples + 1):
            value = rangeMin + (rangeMax - rangeMin) * sample / (samples + 1)
            log.debug('Sampling value %g.', value)
            glyphset = glyphSetFunc(location={axisTag: value})
            designValue = piecewiseLinearMap(value, pins)
            valueMeasurements[designValue] = measureFunc(glyphset, glyphs)
        log.debug('Sampled average value:\n%s', pformat(valueMeasurements))
        measurementValue = {}
        for value in sorted(valueMeasurements):
            measurementValue[valueMeasurements[value]] = value
        out[rangeMin] = targetMin
        outNormalized[normalizedMin] = normalizedTargetMin
        for value in sorted(targetValues):
            t = normalizeFunc(value, rangeMin, rangeMax)
            targetMeasurement = interpolateFunc(t, valueMeasurements[targetMin], valueMeasurements[targetMax])
            targetValue = piecewiseLinearMap(targetMeasurement, measurementValue)
            log.debug('Planned mapping value %g to %g.' % (value, targetValue))
            out[value] = targetValue
            valueNormalized = normalizedMin + (value - rangeMin) / (rangeMax - rangeMin) * (normalizedMax - normalizedMin)
            outNormalized[valueNormalized] = normalizedTargetMin + (targetValue - targetMin) / (targetMax - targetMin) * (normalizedTargetMax - normalizedTargetMin)
        out[rangeMax] = targetMax
        outNormalized[normalizedMax] = normalizedTargetMax
    log.info('Planned mapping for the `%s` axis:\n%s', axisTag, pformat(out))
    log.info('Planned normalized mapping for the `%s` axis:\n%s', axisTag, pformat(outNormalized))
    if all((abs(k - v) < 0.01 for k, v in outNormalized.items())):
        log.info('Detected identity mapping for the `%s` axis. Dropping.', axisTag)
        out = {}
        outNormalized = {}
    return (out, outNormalized)