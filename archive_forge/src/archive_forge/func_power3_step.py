from __future__ import annotations
import operator
from functools import reduce
from math import sqrt
import numpy as np
from scipy.special import erf
def power3_step(xx, edges=None, inverse=False):
    """

    Args:
        xx:
        edges:
        inverse:
    """
    return smoothstep(xx, edges=edges, inverse=inverse)