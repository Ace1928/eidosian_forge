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
def ymax(self):
    """The top edge of the bounding box."""
    return np.max(self.get_points()[:, 1])