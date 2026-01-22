from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
def get_cos_sin(x0, y0, x1, y1):
    dx, dy = (x1 - x0, y1 - y0)
    d = (dx * dx + dy * dy) ** 0.5
    if d == 0:
        return (0.0, 0.0)
    return (dx / d, dy / d)