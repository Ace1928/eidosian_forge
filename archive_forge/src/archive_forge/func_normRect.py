from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def normRect(rect):
    """Normalize a bounding box rectangle.

    This function "turns the rectangle the right way up", so that the following
    holds::

        xMin <= xMax and yMin <= yMax

    Args:
        rect: A bounding rectangle expressed as a tuple
            ``(xMin, yMin, xMax, yMax)``.

    Returns:
        A normalized bounding rectangle.
    """
    xMin, yMin, xMax, yMax = rect
    return (min(xMin, xMax), min(yMin, yMax), max(xMin, xMax), max(yMin, yMax))