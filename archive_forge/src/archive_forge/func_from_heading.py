from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
def from_heading(self, heading: float) -> Vec2:
    """Create a new vector of the same magnitude with the given heading. I.e. Rotate the vector to the heading.

        :parameters:
            `heading` : int or float :
                The angle of the new vector in radians.
        """
    mag = self.__abs__()
    return Vec2(mag * _math.cos(heading), mag * _math.sin(heading))