from typing import Any, Optional, Tuple
import torch
from torch import nn
import torch.distributed as dist
import torch.nn.functional as F
def get_target_nlprob(self, i: torch.Tensor, w: torch.Tensor, target: torch.Tensor, debase_max: torch.Tensor, exp_sums: torch.Tensor) -> torch.Tensor:
    """Get target's negative log probability."""
    target_score = TargetScoreFunction.apply(i, w, target, self)
    prob = (target_score - debase_max).exp() / exp_sums
    if self.log_softmax:
        prob = prob.log()
    return -prob.sum()