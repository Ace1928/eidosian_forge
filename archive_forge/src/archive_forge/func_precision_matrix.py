import math
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
@lazy_property
def precision_matrix(self):
    return torch.cholesky_inverse(self._unbroadcasted_scale_tril).expand(self._batch_shape + self._event_shape + self._event_shape)