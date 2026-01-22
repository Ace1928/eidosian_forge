import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
def vector_angle(u, v):
    d = hypot(*u) * hypot(*v)
    if d == 0:
        return 0
    c = (u[0] * v[0] + u[1] * v[1]) / d
    if c < -1:
        c = -1
    elif c > 1:
        c = 1
    s = u[0] * v[1] - u[1] * v[0]
    return degrees(copysign(acos(c), s))