import numpy as np  # type: ignore
from typing import Tuple, Optional
def calc_angle(v1, v2, v3):
    """Calculate angle method.

    Calculate the angle between 3 vectors
    representing 3 connected points.

    :param v1, v2, v3: the tree points that define the angle
    :type v1, v2, v3: L{Vector}

    :return: angle
    :rtype: float
    """
    v1 = v1 - v2
    v3 = v3 - v2
    return v1.angle(v3)