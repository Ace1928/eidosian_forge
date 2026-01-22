import functools
import math
import numbers
import operator
import weakref
from typing import List
import torch
import torch.nn.functional as F
from torch.distributions import constraints
from torch.distributions.utils import (
from torch.nn.functional import pad, softplus
def with_cache(self, cache_size=1):
    if self._cache_size == cache_size:
        return self
    return CumulativeDistributionTransform(self.distribution, cache_size=cache_size)