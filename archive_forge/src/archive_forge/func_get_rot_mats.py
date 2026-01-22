from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def get_rot_mats(self) -> torch.Tensor:
    """
        Returns the underlying rotation as a rotation matrix tensor.

        Returns:
            The rotation as a rotation matrix tensor
        """
    if self._rot_mats is not None:
        return self._rot_mats
    elif self._quats is not None:
        return quat_to_rot(self._quats)
    else:
        raise ValueError('Both rotations are None')