import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
def convert_quadratic_to_cubic_path(q0, q1, q2):
    """
    Convert a quadratic Bezier curve through q0, q1, q2 to a cubic one.
    """
    c0 = q0
    c1 = (q0[0] + 2 / 3 * (q1[0] - q0[0]), q0[1] + 2 / 3 * (q1[1] - q0[1]))
    c2 = (c1[0] + 1 / 3 * (q2[0] - q0[0]), c1[1] + 1 / 3 * (q2[1] - q0[1]))
    c3 = q2
    return (c0, c1, c2, c3)