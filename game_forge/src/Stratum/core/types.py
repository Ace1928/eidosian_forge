"""
Core type definitions for the Stratum simulation engine.

This module defines small data structures such as 2D vectors and cell
indices used throughout the engine. These simple structures are defined
outside of numpy in order to keep their semantics explicit and to
annotate operations cleanly for type checking. When performance is
critical, numpy arrays are used directly instead of these types.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List
from eidosian_core import eidosian

@dataclass
class Vec2:
    """A simple 2D vector supporting basic arithmetic operations.

    The simulation engine stores momentum using separate x and y arrays,
    but Vec2 is useful for representing small vectors like impulses or
    gradient results in local calculations.

    ``x`` and ``y`` are floating point numbers representing the vector
    components. Use the provided arithmetic helpers below to operate on
    these values safely.
    """
    x: float
    y: float

    def __add__(self, other: Vec2) -> Vec2:
        return Vec2(self.x + other.x, self.y + other.y)

    def __sub__(self, other: Vec2) -> Vec2:
        return Vec2(self.x - other.x, self.y - other.y)

    def __mul__(self, scalar: float) -> Vec2:
        return Vec2(self.x * scalar, self.y * scalar)

    __rmul__ = __mul__

    def __truediv__(self, scalar: float) -> Vec2:
        if scalar == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / scalar, self.y / scalar)

    @eidosian()
    def length(self) -> float:
        return (self.x ** 2 + self.y ** 2) ** 0.5

    @eidosian()
    def normalized(self) -> Vec2:
        l = self.length()
        if l == 0:
            return Vec2(0.0, 0.0)
        return Vec2(self.x / l, self.y / l)


@dataclass(frozen=True)
class Cell:
    """Grid cell coordinate.

    ``i`` and ``j`` are integer indices for the cell position. The
    simulation uses these to index into numpy arrays representing the
    fields. Cells are immutable so they can be used as dictionary keys.
    """
    i: int
    j: int

    @eidosian()
    def neighbors4(self, max_i: int, max_j: int) -> List[Cell]:
        """Return the four orthogonal neighbors within bounds.

        ``max_i`` and ``max_j`` are exclusive upper bounds for the
        indices. Neighbors that fall outside of [0, max_i) × [0, max_j)
        are omitted. This helper is useful for iterating over immediate
        neighbors without duplicating boundary checks throughout the code.
        """
        result: List[Cell] = []
        if self.i > 0:
            result.append(Cell(self.i - 1, self.j))
        if self.i + 1 < max_i:
            result.append(Cell(self.i + 1, self.j))
        if self.j > 0:
            result.append(Cell(self.i, self.j - 1))
        if self.j + 1 < max_j:
            result.append(Cell(self.i, self.j + 1))
        return result


@eidosian()
def dot(a: Vec2, b: Vec2) -> float:
    """Return the dot product of two vectors."""
    return a.x * b.x + a.y * b.y


@eidosian()
def clamp(val: float, min_val: float, max_val: float) -> float:
    """Clamp a value to a specified range."""
    return max(min(val, max_val), min_val)
