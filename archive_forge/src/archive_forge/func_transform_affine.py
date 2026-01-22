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
@_api.rename_parameter('3.8', 'points', 'values')
def transform_affine(self, values):
    if not isinstance(values, np.ndarray):
        _api.warn_external(f'A non-numpy array of type {type(values)} was passed in for transformation, which results in poor performance.')
    return self._transform_affine(values)