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
def get_fully_transformed_path(self):
    """
        Return a fully-transformed copy of the child path.
        """
    self._revalidate()
    return self._transform.transform_path_affine(self._transformed_path)