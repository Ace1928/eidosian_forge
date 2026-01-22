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
def two_sided(null_distribution, observed):
    pvalues_less = less(null_distribution, observed)
    pvalues_greater = greater(null_distribution, observed)
    pvalues = np.minimum(pvalues_less, pvalues_greater) * 2
    return pvalues