from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def insetRect(rect, dx, dy):
    """Inset a bounding box rectangle on all sides.

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.
        dx: Amount to inset the rectangle along the X axis.
        dY: Amount to inset the rectangle along the Y axis.

    Returns:
        An inset bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return (xMin + dx, yMin + dy, xMax - dx, yMax - dy)