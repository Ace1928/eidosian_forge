import logging
from abc import ABC, abstractmethod
from contextlib import ExitStack
from typing import Any, Callable, ContextManager, Dict, Iterable, List, Optional, Tuple, TypeVar, Union
import torch
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins.io.checkpoint_io import CheckpointIO
from lightning_fabric.plugins.io.torch_io import TorchCheckpointIO
from lightning_fabric.plugins.precision import Precision
from lightning_fabric.strategies.launchers.launcher import _Launcher
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.utilities.apply_func import move_data_to_device
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.types import _PATH, Optimizable, ReduceOp, _Stateful
def clip_gradients_value(self, module: torch.nn.Module, optimizer: Optimizer, clip_val: Union[float, int]) -> None:
    """Clip gradients by value."""
    self.precision.unscale_gradients(optimizer)
    parameters = self.precision.main_params(optimizer)
    return torch.nn.utils.clip_grad_value_(parameters, clip_value=clip_val)