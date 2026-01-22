from typing import Dict, Optional, Tuple
import torch
from torch import Tensor
from . import _linalg_utils as _utils
from .overrides import handle_torch_function, has_torch_function
def update_residual(self):
    """Update residual R from A, B, X, E."""
    mm = _utils.matmul
    self.R = mm(self.A, self.X) - mm(self.B, self.X) * self.E