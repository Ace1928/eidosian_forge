from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Type, Union
import torch
from torch.ao.quantization.utils import Pattern
from enum import Enum
def set_fuser_method(self, fuser_method: Callable) -> BackendPatternConfig:
    """
        Set the function that specifies how to fuse this BackendPatternConfig's pattern.

        The first argument of this function should be `is_qat`, and the rest of the arguments
        should be the items in the tuple pattern. The return value of this function should be
        the resulting fused module.

        For example, the fuser method for the pattern `(torch.nn.Linear, torch.nn.ReLU)` can be:

            def fuse_linear_relu(is_qat, linear, relu):
                return torch.ao.nn.intrinsic.LinearReLU(linear, relu)

        For a more complicated example, see https://gist.github.com/jerryzh168/8bea7180a8ba3c279f2c9b050f2a69a6.
        """
    self.fuser_method = fuser_method
    return self