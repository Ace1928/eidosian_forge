from __future__ import annotations
from math import log, pi, sin, sqrt
from ._binary import o8
def sphere_increasing(middle, pos):
    return sqrt(1.0 - (linear(middle, pos) - 1.0) ** 2)