from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
@staticmethod
def from_tensor_7(t: torch.Tensor, normalize_quats: bool=False) -> Rigid:
    if t.shape[-1] != 7:
        raise ValueError('Incorrectly shaped input tensor')
    quats, trans = (t[..., :4], t[..., 4:])
    rots = Rotation(rot_mats=None, quats=quats, normalize_quats=normalize_quats)
    return Rigid(rots, trans)