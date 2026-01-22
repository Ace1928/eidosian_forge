import inspect
import os
from contextlib import contextmanager, nullcontext
from functools import partial
from pathlib import Path
from typing import (
import torch
import torch.nn as nn
from lightning_utilities.core.apply_func import apply_to_collection
from lightning_utilities.core.overrides import is_overridden
from torch import Tensor
from torch.optim import Optimizer
from torch.utils.data import BatchSampler, DataLoader, DistributedSampler, RandomSampler, SequentialSampler
from lightning_fabric.accelerators.accelerator import Accelerator
from lightning_fabric.connector import _PLUGIN_INPUT, _PRECISION_INPUT, _Connector, _is_using_cli
from lightning_fabric.loggers import Logger
from lightning_fabric.plugins import Precision  # avoid circular imports: # isort: split
from lightning_fabric.strategies import (
from lightning_fabric.strategies.fsdp import _has_meta_device_parameters
from lightning_fabric.strategies.launchers import _MultiProcessingLauncher, _XLALauncher
from lightning_fabric.strategies.strategy import TBroadcast, _Sharded
from lightning_fabric.utilities import move_data_to_device
from lightning_fabric.utilities.apply_func import convert_tensors_to_scalars, convert_to_tensors
from lightning_fabric.utilities.data import (
from lightning_fabric.utilities.device_dtype_mixin import _update_properties
from lightning_fabric.utilities.distributed import DistributedSamplerWrapper
from lightning_fabric.utilities.imports import _TORCH_GREATER_EQUAL_2_0
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_warn
from lightning_fabric.utilities.registry import _load_external_callbacks
from lightning_fabric.utilities.seed import seed_everything
from lightning_fabric.utilities.types import ReduceOp
from lightning_fabric.utilities.warnings import PossibleUserWarning
from lightning_fabric.wrappers import (
def setup_dataloaders(self, *dataloaders: DataLoader, use_distributed_sampler: bool=True, move_to_device: bool=True) -> Union[DataLoader, List[DataLoader]]:
    """Set up one or multiple dataloaders for accelerated training. If you need different settings for each
        dataloader, call this method individually for each one.

        Args:
            *dataloaders: A single dataloader or a sequence of dataloaders.
            use_distributed_sampler: If set ``True`` (default), automatically wraps or replaces the sampler on the
                dataloader(s) for distributed training. If you have a custom sampler defined, set this argument
                to ``False``.
            move_to_device: If set ``True`` (default), moves the data returned by the dataloader(s) automatically to
                the correct device. Set this to ``False`` and alternatively use :meth:`to_device` manually on the
                returned data.

        Returns:
            The wrapped dataloaders, in the same order they were passed in.

        """
    self._validate_setup_dataloaders(dataloaders)
    dataloaders = [self._setup_dataloader(dataloader, use_distributed_sampler=use_distributed_sampler, move_to_device=move_to_device) for dataloader in dataloaders]
    dataloaders = dataloaders[0] if len(dataloaders) == 1 else dataloaders
    return dataloaders