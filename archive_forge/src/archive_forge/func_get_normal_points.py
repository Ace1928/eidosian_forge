from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def get_normal_points(cx, cy, cos_t, sin_t, length):
    """
    For a line passing through (*cx*, *cy*) and having an angle *t*, return
    locations of the two points located along its perpendicular line at the
    distance of *length*.
    """
    if length == 0.0:
        return (cx, cy, cx, cy)
    cos_t1, sin_t1 = (sin_t, -cos_t)
    cos_t2, sin_t2 = (-sin_t, cos_t)
    x1, y1 = (length * cos_t1 + cx, length * sin_t1 + cy)
    x2, y2 = (length * cos_t2 + cx, length * sin_t2 + cy)
    return (x1, y1, x2, y2)