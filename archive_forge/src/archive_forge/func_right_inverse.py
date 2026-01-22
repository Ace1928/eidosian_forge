from enum import Enum, auto
import torch
from torch import Tensor
from ..utils import parametrize
from ..modules import Module
from .. import functional as F
from typing import Optional
def right_inverse(self, value: torch.Tensor) -> torch.Tensor:
    return value