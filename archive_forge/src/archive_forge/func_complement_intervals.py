from typing import List, Sequence, Tuple
import itertools
import numpy as np
import pytest
import sympy
import cirq
def complement_intervals(intervals: Sequence[Tuple[float, float]]) -> Sequence[Tuple[float, float]]:
    """Computes complement of union of intervals in [0, 2]."""
    complements: List[Tuple[float, float]] = []
    a = 0.0
    for b, c in intervals:
        complements.append((a, b))
        a = c
    complements.append((a, 2.0))
    return tuple(((a, b) for a, b in complements if a < b))