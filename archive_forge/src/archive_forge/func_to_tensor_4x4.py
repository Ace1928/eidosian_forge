from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def to_tensor_4x4(self) -> torch.Tensor:
    """
        Converts a transformation to a homogenous transformation tensor.

        Returns:
            A [*, 4, 4] homogenous transformation tensor
        """
    tensor = self._trans.new_zeros((*self.shape, 4, 4))
    tensor[..., :3, :3] = self._rots.get_rot_mats()
    tensor[..., :3, 3] = self._trans
    tensor[..., 3, 3] = 1
    return tensor