from functools import update_wrapper
from numbers import Number
from typing import Any, Dict
import torch
import torch.nn.functional as F
from torch.overrides import is_tensor_like
def logits_to_probs(logits, is_binary=False):
    """
    Converts a tensor of logits into probabilities. Note that for the
    binary case, each value denotes log odds, whereas for the
    multi-dimensional case, the values along the last dimension denote
    the log probabilities (possibly unnormalized) of the events.
    """
    if is_binary:
        return torch.sigmoid(logits)
    return F.softmax(logits, dim=-1)