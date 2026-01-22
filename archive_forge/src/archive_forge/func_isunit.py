import math
import numpy as np
from .casting import sctypes
def isunit(q):
    """Return True is this is very nearly a unit quaternion"""
    return np.allclose(norm(q), 1)