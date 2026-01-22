from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def invert_quat(quat: torch.Tensor) -> torch.Tensor:
    quat_prime = quat.clone()
    quat_prime[..., 1:] *= -1
    inv = quat_prime / torch.sum(quat ** 2, dim=-1, keepdim=True)
    return inv