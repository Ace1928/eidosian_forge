from collections import namedtuple
from dataclasses import dataclass
from math import comb
import numpy as np
import warnings
from itertools import combinations
import scipy.stats
from scipy.optimize import shgo
from . import distributions
from ._common import ConfidenceInterval
from ._continuous_distns import chi2, norm
from scipy.special import gamma, kv, gammaln
from scipy.fft import ifft
from ._stats_pythran import _a_ij_Aij_Dij2
from ._stats_pythran import (
from ._axis_nan_policy import _axis_nan_policy_factory
from scipy.stats import _stats_py
def term(x, k):
    u = np.exp(gammaln(k + 0.5) - gammaln(k + 1)) / (np.pi ** 1.5 * np.sqrt(x))
    y = 4 * k + 1
    q = y ** 2 / (16 * x)
    b = kv(0.25, q)
    return u * np.sqrt(y) * np.exp(-q) * b