import torch
from torch.nn.functional import normalize
from typing import Any, Optional, TypeVar
from ..modules import Module
def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
    weight_mat = weight
    if self.dim != 0:
        weight_mat = weight_mat.permute(self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim])
    height = weight_mat.size(0)
    return weight_mat.reshape(height, -1)