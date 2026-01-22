from fontTools.misc.roundTools import otRound
from fontTools.misc.vector import Vector as _Vector
import math
import warnings
def unionRect(rect1, rect2):
    """Determine union of bounding rectangles.

    Args:
        rect1: First bounding rectangle, expressed as tuples
            ``(xMin, yMin, xMax, yMax)``.
        rect2: Second bounding rectangle.

    Returns:
        The smallest rectangle in which both input rectangles are fully
        enclosed.
    """
    xMin1, yMin1, xMax1, yMax1 = rect1
    xMin2, yMin2, xMax2, yMax2 = rect2
    xMin, yMin, xMax, yMax = (min(xMin1, xMin2), min(yMin1, yMin2), max(xMax1, xMax2), max(yMax1, yMax2))
    return (xMin, yMin, xMax, yMax)