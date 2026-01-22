import copy
import functools
import textwrap
import weakref
import math
import numpy as np
from numpy.linalg import inv
from matplotlib import _api
from matplotlib._path import (
from .path import Path
@property
def ymin(self):
    """The bottom edge of the bounding box."""
    return np.min(self.get_points()[:, 1])