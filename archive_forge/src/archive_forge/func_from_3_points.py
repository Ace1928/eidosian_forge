from __future__ import annotations
from functools import lru_cache
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple
import numpy as np
import torch
@staticmethod
def from_3_points(p_neg_x_axis: torch.Tensor, origin: torch.Tensor, p_xy_plane: torch.Tensor, eps: float=1e-08) -> Rigid:
    """
        Implements algorithm 21. Constructs transformations from sets of 3 points using the Gram-Schmidt algorithm.

        Args:
            p_neg_x_axis: [*, 3] coordinates
            origin: [*, 3] coordinates used as frame origins
            p_xy_plane: [*, 3] coordinates
            eps: Small epsilon value
        Returns:
            A transformation object of shape [*]
        """
    p_neg_x_axis_unbound = torch.unbind(p_neg_x_axis, dim=-1)
    origin_unbound = torch.unbind(origin, dim=-1)
    p_xy_plane_unbound = torch.unbind(p_xy_plane, dim=-1)
    e0 = [c1 - c2 for c1, c2 in zip(origin_unbound, p_neg_x_axis_unbound)]
    e1 = [c1 - c2 for c1, c2 in zip(p_xy_plane_unbound, origin_unbound)]
    denom = torch.sqrt(sum((c * c for c in e0)) + eps * torch.ones_like(e0[0]))
    e0 = [c / denom for c in e0]
    dot = sum((c1 * c2 for c1, c2 in zip(e0, e1)))
    e1 = [c2 - c1 * dot for c1, c2 in zip(e0, e1)]
    denom = torch.sqrt(sum((c * c for c in e1)) + eps * torch.ones_like(e1[0]))
    e1 = [c / denom for c in e1]
    e2 = [e0[1] * e1[2] - e0[2] * e1[1], e0[2] * e1[0] - e0[0] * e1[2], e0[0] * e1[1] - e0[1] * e1[0]]
    rots = torch.stack([c for tup in zip(e0, e1, e2) for c in tup], dim=-1)
    rots = rots.reshape(rots.shape[:-1] + (3, 3))
    rot_obj = Rotation(rot_mats=rots, quats=None)
    return Rigid(rot_obj, torch.stack(origin_unbound, dim=-1))