from typing import Callable, Dict, Optional, Tuple
import torch
from torch import nn
from torch.distributions import (
@classmethod
def domain_map(cls, total_count: torch.Tensor, logits: torch.Tensor):
    total_count = cls.squareplus(total_count)
    return (total_count.squeeze(-1), logits.squeeze(-1))