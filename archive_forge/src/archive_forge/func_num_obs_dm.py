import math
import warnings
import numpy as np
import dataclasses
from typing import Optional, Callable
from functools import partial
from scipy._lib._util import _asarray_validated
from . import _distance_wrap
from . import _hausdorff
from ..linalg import norm
from ..special import rel_entr
from . import _distance_pybind
def num_obs_dm(d):
    """
    Return the number of original observations that correspond to a
    square, redundant distance matrix.

    Parameters
    ----------
    d : array_like
        The target distance matrix.

    Returns
    -------
    num_obs_dm : int
        The number of observations in the redundant distance matrix.

    Examples
    --------
    Find the number of original observations corresponding
    to a square redundant distance matrix d.
    
    >>> from scipy.spatial.distance import num_obs_dm
    >>> d = [[0, 100, 200], [100, 0, 150], [200, 150, 0]]
    >>> num_obs_dm(d)
    3
    """
    d = np.asarray(d, order='c')
    is_valid_dm(d, tol=np.inf, throw=True, name='d')
    return d.shape[0]