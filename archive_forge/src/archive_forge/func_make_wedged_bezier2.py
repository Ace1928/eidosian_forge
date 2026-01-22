from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def make_wedged_bezier2(bezier2, width, w1=1.0, wm=0.5, w2=0.0):
    """
    Being similar to `get_parallels`, returns control points of two quadratic
    Bézier lines having a width roughly parallel to given one separated by
    *width*.
    """
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c3x, c3y = bezier2[2]
    cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
    cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c3x, c3y)
    c1x_left, c1y_left, c1x_right, c1y_right = get_normal_points(c1x, c1y, cos_t1, sin_t1, width * w1)
    c3x_left, c3y_left, c3x_right, c3y_right = get_normal_points(c3x, c3y, cos_t2, sin_t2, width * w2)
    c12x, c12y = ((c1x + cmx) * 0.5, (c1y + cmy) * 0.5)
    c23x, c23y = ((cmx + c3x) * 0.5, (cmy + c3y) * 0.5)
    c123x, c123y = ((c12x + c23x) * 0.5, (c12y + c23y) * 0.5)
    cos_t123, sin_t123 = get_cos_sin(c12x, c12y, c23x, c23y)
    c123x_left, c123y_left, c123x_right, c123y_right = get_normal_points(c123x, c123y, cos_t123, sin_t123, width * wm)
    path_left = find_control_points(c1x_left, c1y_left, c123x_left, c123y_left, c3x_left, c3y_left)
    path_right = find_control_points(c1x_right, c1y_right, c123x_right, c123y_right, c3x_right, c3y_right)
    return (path_left, path_right)