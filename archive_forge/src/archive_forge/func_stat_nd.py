from __future__ import annotations
import warnings
import numpy as np
from itertools import combinations, permutations, product
from collections.abc import Sequence
import inspect
from scipy._lib._util import check_random_state, _rename_parameter
from scipy.special import ndtr, ndtri, comb, factorial
from scipy._lib._util import rng_integers
from dataclasses import dataclass
from ._common import ConfidenceInterval
from ._axis_nan_policy import _broadcast_concatenate, _broadcast_arrays
from ._warnings_errors import DegenerateDataWarning
def stat_nd(*data, axis=0):
    lengths = [sample.shape[axis] for sample in data]
    split_indices = np.cumsum(lengths)[:-1]
    z = _broadcast_concatenate(data, axis)
    z = np.moveaxis(z, axis, 0)

    def stat_1d(z):
        data = np.split(z, split_indices)
        return statistic(*data)
    return np.apply_along_axis(stat_1d, 0, z)[()]