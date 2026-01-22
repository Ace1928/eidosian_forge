from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def scale_and_clamp(xx, edge0, edge1, clamp0, clamp1):
    """

    Args:
        xx:
        edge0:
        edge1:
        clamp0:
        clamp1:
    """
    return np.clip((xx - edge0) / (edge1 - edge0), clamp0, clamp1)