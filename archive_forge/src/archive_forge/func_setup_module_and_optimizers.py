import shutil
from contextlib import ExitStack
from datetime import timedelta
from functools import partial
from pathlib import Path
from typing import (
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module, Parameter
from torch.optim import Optimizer
from typing_extensions import TypeGuard, override
from lightning_fabric.accelerators import Accelerator
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment, Precision
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.plugins.precision.fsdp import FSDPPrecision
from lightning_fabric.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from lightning_fabric.strategies.parallel import ParallelStrategy
from lightning_fabric.strategies.registry import _StrategyRegistry
from lightning_fabric.strategies.strategy import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _METADATA_FILENAME, _lazy_load, _materialize_tensors, _move_state_into
from lightning_fabric.utilities.rank_zero import rank_zero_deprecation, rank_zero_only, rank_zero_warn
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, _Stateful
@override
def setup_module_and_optimizers(self, module: Module, optimizers: List[Optimizer]) -> Tuple[Module, List[Optimizer]]:
    """Wraps the model into a :class:`~torch.distributed.fsdp.fully_sharded_data_parallel.FullyShardedDataParallel`
        module and sets `use_orig_params=True` to keep the reference to the original parameters in the optimizer."""
    if not _TORCH_GREATER_EQUAL_2_0:
        raise NotImplementedError(f'The `{type(self).__name__}` does not support the joint setup of module and optimizer(s). Please do it in this order: Create the model, call `setup_module`, create the optimizer, call `setup_optimizer`.')
    use_orig_params = self._fsdp_kwargs.get('use_orig_params')
    if use_orig_params is False:
        raise ValueError(f'You set `{type(self).__name__}(use_orig_params=False)` but this is not supported when setting the model and optimizer up jointly. Either set it to `True` or set the objects up in this order: Create the model, call `setup_module`, create the optimizer, call `setup_optimizer`.')
    module = self.setup_module(module)
    return (module, optimizers)