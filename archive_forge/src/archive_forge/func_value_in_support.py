from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
@property
def value_in_support(self) -> float:
    """
        A float that will have a valid numeric value when computing the log-loss of the corresponding distribution. By
        default 0.0. This value will be used when padding data series.
        """
    return 0.0