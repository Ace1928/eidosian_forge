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
def sanitizeWeight(userTriple, designTriple, pins, measurements):
    """Sanitize the weight axis limits."""
    if len(set(userTriple)) < 3:
        return True
    minVal, defaultVal, maxVal = (measurements[designTriple[0]], measurements[designTriple[1]], measurements[designTriple[2]])
    logMin = math.log(minVal)
    logDefault = math.log(defaultVal)
    logMax = math.log(maxVal)
    t = (userTriple[1] - userTriple[0]) / (userTriple[2] - userTriple[0])
    y = math.exp(logMin + t * (logMax - logMin))
    t = (y - minVal) / (maxVal - minVal)
    calculatedDefaultVal = userTriple[0] + t * (userTriple[2] - userTriple[0])
    log.info('Original weight axis limits: %g:%g:%g', *userTriple)
    log.info('Calculated weight axis limits: %g:%g:%g', userTriple[0], calculatedDefaultVal, userTriple[2])
    if abs(calculatedDefaultVal - userTriple[1]) / userTriple[1] > 0.05:
        log.warning('Calculated weight axis default does not match user input.')
        log.warning('  Current weight axis limits: %g:%g:%g', *userTriple)
        log.warning('  Suggested weight axis limits, changing default: %g:%g:%g', userTriple[0], calculatedDefaultVal, userTriple[2])
        t = (userTriple[2] - userTriple[0]) / (userTriple[1] - userTriple[0])
        y = math.exp(logMin + t * (logDefault - logMin))
        t = (y - minVal) / (defaultVal - minVal)
        calculatedMaxVal = userTriple[0] + t * (userTriple[1] - userTriple[0])
        log.warning('  Suggested weight axis limits, changing maximum: %g:%g:%g', userTriple[0], userTriple[1], calculatedMaxVal)
        t = (userTriple[0] - userTriple[2]) / (userTriple[1] - userTriple[2])
        y = math.exp(logMax + t * (logDefault - logMax))
        t = (y - maxVal) / (defaultVal - maxVal)
        calculatedMinVal = userTriple[2] + t * (userTriple[1] - userTriple[2])
        log.warning('  Suggested weight axis limits, changing minimum: %g:%g:%g', calculatedMinVal, userTriple[1], userTriple[2])
        return False
    return True