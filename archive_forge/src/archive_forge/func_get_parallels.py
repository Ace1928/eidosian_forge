from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def get_parallels(bezier2, width):
    """
    Given the quadratic Bézier control points *bezier2*, returns
    control points of quadratic Bézier lines roughly parallel to given
    one separated by *width*.
    """
    c1x, c1y = bezier2[0]
    cmx, cmy = bezier2[1]
    c2x, c2y = bezier2[2]
    parallel_test = check_if_parallel(c1x - cmx, c1y - cmy, cmx - c2x, cmy - c2y)
    if parallel_test == -1:
        _api.warn_external('Lines do not intersect. A straight line is used instead.')
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, c2x, c2y)
        cos_t2, sin_t2 = (cos_t1, sin_t1)
    else:
        cos_t1, sin_t1 = get_cos_sin(c1x, c1y, cmx, cmy)
        cos_t2, sin_t2 = get_cos_sin(cmx, cmy, c2x, c2y)
    c1x_left, c1y_left, c1x_right, c1y_right = get_normal_points(c1x, c1y, cos_t1, sin_t1, width)
    c2x_left, c2y_left, c2x_right, c2y_right = get_normal_points(c2x, c2y, cos_t2, sin_t2, width)
    try:
        cmx_left, cmy_left = get_intersection(c1x_left, c1y_left, cos_t1, sin_t1, c2x_left, c2y_left, cos_t2, sin_t2)
        cmx_right, cmy_right = get_intersection(c1x_right, c1y_right, cos_t1, sin_t1, c2x_right, c2y_right, cos_t2, sin_t2)
    except ValueError:
        cmx_left, cmy_left = (0.5 * (c1x_left + c2x_left), 0.5 * (c1y_left + c2y_left))
        cmx_right, cmy_right = (0.5 * (c1x_right + c2x_right), 0.5 * (c1y_right + c2y_right))
    path_left = [(c1x_left, c1y_left), (cmx_left, cmy_left), (c2x_left, c2y_left)]
    path_right = [(c1x_right, c1y_right), (cmx_right, cmy_right), (c2x_right, c2y_right)]
    return (path_left, path_right)