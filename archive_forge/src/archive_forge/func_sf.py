import numpy as np
from collections import namedtuple
from scipy import special
from scipy import stats
from ._axis_nan_policy import _axis_nan_policy_factory
def sf(self, k, m, n):
    """Survival function"""
    k = m * n - k
    return self.cdf(k, m, n)