import math
from typing import NamedTuple
from dataclasses import dataclass
def transformVectors(self, vectors):
    """Transform a list of (dx, dy) vector, treating translation as zero.

        :Example:
                >>> t = Transform(2, 0, 0, 2, 10, 20)
                >>> t.transformVectors([(3, -4), (5, -6)])
                [(6, -8), (10, -12)]
                >>>
        """
    xx, xy, yx, yy = self[:4]
    return [(xx * dx + yx * dy, xy * dx + yy * dy) for dx, dy in vectors]