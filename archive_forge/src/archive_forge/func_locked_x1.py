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
@locked_x1.setter
def locked_x1(self, x1):
    self._locked_points.mask[1, 0] = x1 is None
    self._locked_points.data[1, 0] = x1
    self.invalidate()