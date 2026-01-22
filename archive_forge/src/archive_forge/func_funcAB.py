from __future__ import annotations
import inspect
from dataclasses import dataclass
from typing import (
import numpy as np
from scipy.stats._common import ConfidenceInterval
from scipy.stats._qmc import check_random_state
from scipy.stats._resampling import BootstrapResult
from scipy.stats import qmc, bootstrap
def funcAB(AB):
    d, d, n = AB.shape
    AB = np.moveaxis(AB, 0, -1).reshape(d, n * d)
    f_AB = wrapped_func(AB)
    return np.moveaxis(f_AB.reshape((-1, n, d)), -1, 0)