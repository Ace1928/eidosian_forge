from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
def from_magnitude(self, magnitude: float) -> Vec3:
    """Create a new Vector of the given magnitude by normalizing,
        then scaling the vector. The rotation remains unchanged.
        """
    return self.normalize() * magnitude