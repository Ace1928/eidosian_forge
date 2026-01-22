from typing import cast, List, Optional, Callable, Tuple
import torch
from torch import nn, Tensor
from torch.nn.utils import parametrize
from torch.nn.utils.parametrize import ParametrizationList
from .parametrization import FakeStructuredSparsity, BiasHook
def prune_conv2d_pool_activation_conv2d(c1: nn.Conv2d, pool: nn.Module, activation: Optional[Callable[[Tensor], Tensor]], c2: nn.Conv2d) -> None:
    prune_conv2d_activation_conv2d(c1, activation, c2)