import numpy as np  # type: ignore
from typing import Tuple, Optional
def vector_to_axis(line, point):
    """Vector to axis method.

    Return the vector between a point and
    the closest point on a line (ie. the perpendicular
    projection of the point on the line).

    :type line: L{Vector}
    :param line: vector defining a line

    :type point: L{Vector}
    :param point: vector defining the point
    """
    line = line.normalized()
    np = point.norm()
    angle = line.angle(point)
    return point - line ** (np * np.cos(angle))