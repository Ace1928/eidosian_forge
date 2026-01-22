from typing import Dict
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _sum_rightmost
@property
def has_enumerate_support(self):
    if self.reinterpreted_batch_ndims > 0:
        return False
    return self.base_dist.has_enumerate_support