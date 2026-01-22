from typing import Optional
import torch
from torch import nn
from .. import _is_triton_available
def increment_and_forward_(self, x: torch.Tensor, y: torch.Tensor):
    """
        An addition fused with forward.

            z = layer.increment_and_forward_(x, y)

        is equivalent to

            x += y
            z = layer(x)
        """
    return rms_norm_add(x, y, self.weight, self.eps)