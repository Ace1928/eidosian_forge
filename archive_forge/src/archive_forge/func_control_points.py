from functools import lru_cache
import math
import warnings
import numpy as np
from matplotlib import _api
@property
def control_points(self):
    """The control points of the curve."""
    return self._cpoints