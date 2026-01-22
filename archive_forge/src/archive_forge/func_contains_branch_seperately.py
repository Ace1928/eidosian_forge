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
def contains_branch_seperately(self, transform):
    return (self._x.contains_branch(transform), self._y.contains_branch(transform))