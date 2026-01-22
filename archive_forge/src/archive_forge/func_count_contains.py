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
def count_contains(self, vertices):
    """
        Count the number of vertices contained in the `Bbox`.
        Any vertices with a non-finite x or y value are ignored.

        Parameters
        ----------
        vertices : (N, 2) array
        """
    if len(vertices) == 0:
        return 0
    vertices = np.asarray(vertices)
    with np.errstate(invalid='ignore'):
        return ((self.min < vertices) & (vertices < self.max)).all(axis=1).sum()