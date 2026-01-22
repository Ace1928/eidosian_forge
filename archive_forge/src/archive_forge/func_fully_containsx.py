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
def fully_containsx(self, x):
    """
        Return whether *x* is in the open (:attr:`x0`, :attr:`x1`) interval.
        """
    x0, x1 = self.intervalx
    return x0 < x < x1 or x0 > x > x1