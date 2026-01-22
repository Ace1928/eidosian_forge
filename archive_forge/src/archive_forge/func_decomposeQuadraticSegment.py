from typing import Tuple, Dict
from fontTools.misc.loggingTools import LogMixin
from fontTools.misc.transform import DecomposedTransform, Identity
def decomposeQuadraticSegment(points):
    """Split the quadratic curve segment described by 'points' into a list
    of "atomic" quadratic segments. The 'points' argument must be a sequence
    with length 2 or greater, containing (x, y) coordinates. The last point
    is the destination on-curve point, the rest of the points are off-curve
    points. The start point should not be supplied.

    This function returns a list of (pt1, pt2) tuples, which each specify a
    plain quadratic bezier segment.
    """
    n = len(points) - 1
    assert n > 0
    quadSegments = []
    for i in range(n - 1):
        x, y = points[i]
        nx, ny = points[i + 1]
        impliedPt = (0.5 * (x + nx), 0.5 * (y + ny))
        quadSegments.append((points[i], impliedPt))
    quadSegments.append((points[-2], points[-1]))
    return quadSegments