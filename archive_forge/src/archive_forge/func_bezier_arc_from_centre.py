import re, copy
from math import acos, ceil, copysign, cos, degrees, fabs, hypot, radians, sin, sqrt
from .shapes import Group, mmult, rotate, translate, transformPoint, Path, FILL_EVEN_ODD, _CLOSEPATH, UserNode
def bezier_arc_from_centre(cx, cy, rx, ry, start_ang=0, extent=90):
    if abs(extent) <= 90:
        nfrag = 1
        frag_angle = extent
    else:
        nfrag = ceil(abs(extent) / 90)
        frag_angle = extent / nfrag
    if frag_angle == 0:
        return []
    frag_rad = radians(frag_angle)
    half_rad = frag_rad * 0.5
    kappa = abs(4 / 3 * (1 - cos(half_rad)) / sin(half_rad))
    if frag_angle < 0:
        kappa = -kappa
    point_list = []
    theta1 = radians(start_ang)
    start_rad = theta1 + frag_rad
    c1 = cos(theta1)
    s1 = sin(theta1)
    for i in range(nfrag):
        c0 = c1
        s0 = s1
        theta1 = start_rad + i * frag_rad
        c1 = cos(theta1)
        s1 = sin(theta1)
        point_list.append((cx + rx * c0, cy - ry * s0, cx + rx * (c0 - kappa * s0), cy - ry * (s0 + kappa * c0), cx + rx * (c1 + kappa * s1), cy - ry * (s1 - kappa * c1), cx + rx * c1, cy - ry * s1))
    return point_list