import torch
from torch import inf
from torch.distributions import Categorical, constraints
from torch.distributions.binomial import Binomial
from torch.distributions.distribution import Distribution
from torch.distributions.utils import broadcast_all
@property
def param_shape(self):
    return self._categorical.param_shape