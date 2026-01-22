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
def fully_containsy(self, y):
    """
        Return whether *y* is in the open (:attr:`y0`, :attr:`y1`) interval.
        """
    y0, y1 = self.intervaly
    return y0 < y < y1 or y0 > y > y1