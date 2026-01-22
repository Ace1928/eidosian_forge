from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
def get_quats(self) -> torch.Tensor:
    """
        Returns the underlying rotation as a quaternion tensor.

        Depending on whether the Rotation was initialized with a quaternion, this function may call torch.linalg.eigh.

        Returns:
            The rotation as a quaternion tensor.
        """
    if self._rot_mats is not None:
        return rot_to_quat(self._rot_mats)
    elif self._quats is not None:
        return self._quats
    else:
        raise ValueError('Both rotations are None')