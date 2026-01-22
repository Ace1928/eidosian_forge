import math
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.utils import _standard_normal, lazy_property
@lazy_property
def scale_tril(self):
    return self._unbroadcasted_scale_tril.expand(self._batch_shape + self._event_shape + self._event_shape)