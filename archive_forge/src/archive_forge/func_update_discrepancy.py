from __future__ import annotations
import copy
import math
import numbers
import os
import warnings
from abc import ABC, abstractmethod
from functools import partial
from typing import (
import numpy as np
import scipy.stats as stats
from scipy._lib._util import rng_integers, _rng_spawn
from scipy.sparse.csgraph import minimum_spanning_tree
from scipy.spatial import distance, Voronoi
from scipy.special import gammainc
from ._sobol import (
from ._qmc_cy import (
def update_discrepancy(x_new: npt.ArrayLike, sample: npt.ArrayLike, initial_disc: DecimalNumber) -> float:
    """Update the centered discrepancy with a new sample.

    Parameters
    ----------
    x_new : array_like (1, d)
        The new sample to add in `sample`.
    sample : array_like (n, d)
        The initial sample.
    initial_disc : float
        Centered discrepancy of the `sample`.

    Returns
    -------
    discrepancy : float
        Centered discrepancy of the sample composed of `x_new` and `sample`.

    Examples
    --------
    We can also compute iteratively the discrepancy by using
    ``iterative=True``.

    >>> import numpy as np
    >>> from scipy.stats import qmc
    >>> space = np.array([[1, 3], [2, 6], [3, 2], [4, 5], [5, 1], [6, 4]])
    >>> l_bounds = [0.5, 0.5]
    >>> u_bounds = [6.5, 6.5]
    >>> space = qmc.scale(space, l_bounds, u_bounds, reverse=True)
    >>> disc_init = qmc.discrepancy(space[:-1], iterative=True)
    >>> disc_init
    0.04769081147119336
    >>> qmc.update_discrepancy(space[-1], space[:-1], disc_init)
    0.008142039609053513

    """
    sample = np.asarray(sample, dtype=np.float64, order='C')
    x_new = np.asarray(x_new, dtype=np.float64, order='C')
    if not sample.ndim == 2:
        raise ValueError('Sample is not a 2D array')
    if sample.max() > 1.0 or sample.min() < 0.0:
        raise ValueError('Sample is not in unit hypercube')
    if not x_new.ndim == 1:
        raise ValueError('x_new is not a 1D array')
    if not (np.all(x_new >= 0) and np.all(x_new <= 1)):
        raise ValueError('x_new is not in unit hypercube')
    if x_new.shape[0] != sample.shape[1]:
        raise ValueError('x_new and sample must be broadcastable')
    return _cy_wrapper_update_discrepancy(x_new, sample, initial_disc)