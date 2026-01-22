from __future__ import annotations
import math as _math
import typing as _typing
import warnings as _warnings
from operator import mul as _mul
from collections.abc import Iterable as _Iterable
from collections.abc import Iterator as _Iterator
@classmethod
def from_rotation(cls, angle: float, vector: Vec3) -> Mat4:
    """Create a rotation matrix from an angle and Vec3."""
    return cls().rotate(angle, vector)