from typing import Dict
import torch
from torch.distributions import constraints
from torch.distributions.distribution import Distribution
from torch.distributions.independent import Independent
from torch.distributions.transforms import ComposeTransform, Transform
from torch.distributions.utils import _sum_rightmost

        Computes the inverse cumulative distribution function using
        transform(s) and computing the score of the base distribution.
        