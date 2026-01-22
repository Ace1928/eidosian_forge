from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
@staticmethod
def computeAxisRanges(locations):
    axisRanges = {}
    allAxes = {axis for loc in locations for axis in loc.keys()}
    for loc in locations:
        for axis in allAxes:
            value = loc.get(axis, 0)
            axisMin, axisMax = axisRanges.get(axis, (value, value))
            axisRanges[axis] = (min(value, axisMin), max(value, axisMax))
    return axisRanges