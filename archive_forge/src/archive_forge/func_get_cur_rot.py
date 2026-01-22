from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def get_cur_rot(self) -> torch.Tensor:
    """
        Return the underlying rotation in its current form

        Returns:
            The stored rotation
        """
    if self._rot_mats is not None:
        return self._rot_mats
    elif self._quats is not None:
        return self._quats
    else:
        raise ValueError('Both rotations are None')