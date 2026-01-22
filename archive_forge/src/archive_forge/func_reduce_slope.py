from .simplex import *
from .tetrahedron import Tetrahedron
import sys
from .linalg import Vector, Matrix
def reduce_slope(slope):
    a, b = slope
    if a == b == 0:
        return (slope, 0)
    g = gcd(a, b)
    a, b = (a / g, b / g)
    return ((a, b), g)