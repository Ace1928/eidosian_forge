from fontTools.misc.roundTools import noRound
from .errors import VariationModelError
@staticmethod
def getMasterLocationsSortKeyFunc(locations, axisOrder=[]):
    if {} not in locations:
        raise VariationModelError('Base master not found.')
    axisPoints = {}
    for loc in locations:
        if len(loc) != 1:
            continue
        axis = next(iter(loc))
        value = loc[axis]
        if axis not in axisPoints:
            axisPoints[axis] = {0.0}
        assert value not in axisPoints[axis], 'Value "%s" in axisPoints["%s"] -->  %s' % (value, axis, axisPoints)
        axisPoints[axis].add(value)

    def getKey(axisPoints, axisOrder):

        def sign(v):
            return -1 if v < 0 else +1 if v > 0 else 0

        def key(loc):
            rank = len(loc)
            onPointAxes = [axis for axis, value in loc.items() if axis in axisPoints and value in axisPoints[axis]]
            orderedAxes = [axis for axis in axisOrder if axis in loc]
            orderedAxes.extend([axis for axis in sorted(loc.keys()) if axis not in axisOrder])
            return (rank, -len(onPointAxes), tuple((axisOrder.index(axis) if axis in axisOrder else 65536 for axis in orderedAxes)), tuple(orderedAxes), tuple((sign(loc[axis]) for axis in orderedAxes)), tuple((abs(loc[axis]) for axis in orderedAxes)))
        return key
    ret = getKey(axisPoints, axisOrder)
    return ret