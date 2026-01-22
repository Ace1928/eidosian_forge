import logging
import shutil
from contextlib import contextmanager, nullcontext
from datetime import timedelta
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Dict, Generator, List, Literal, Mapping, Optional, Set, Type, Union
import torch
from lightning_utilities.core.rank_zero import rank_zero_only as utils_rank_zero_only
from torch import Tensor
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import CheckpointIO, ClusterEnvironment
from lightning_fabric.plugins.collectives.torch_collective import default_pg_timeout
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.fsdp import (
from lightning_fabric.utilities.distributed import (
from lightning_fabric.utilities.distributed import group as _group
from lightning_fabric.utilities.imports import (
from lightning_fabric.utilities.init import _EmptyInit
from lightning_fabric.utilities.load import _lazy_load, _materialize_tensors
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, ReduceOp
from pytorch_lightning.core.optimizer import LightningOptimizer
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.plugins.precision.fsdp import FSDPPrecision
from pytorch_lightning.strategies.launchers.subprocess_script import _SubprocessScriptLauncher
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import TBroadcast
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_only, rank_zero_warn
@override
def optimizer_state(self, optimizer: Optimizer) -> Dict[str, Tensor]:
    if not _TORCH_GREATER_EQUAL_2_0:
        rank_zero_warn('FSDP in Lightning with PyTorch < 2.0 does not support saving the optimizer state.')
        return {}
    from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
    from torch.distributed.fsdp import OptimStateKeyType
    if isinstance(optimizer, LightningOptimizer):
        optimizer = optimizer._optimizer
    assert self.model is not None
    if self._state_dict_type == 'sharded':
        with _get_sharded_state_dict_context(self.model):
            return FSDP.optim_state_dict(self.model, optimizer)
    elif self._state_dict_type == 'full':
        with _get_full_state_dict_context(self.model, world_size=self.world_size):
            state_dict = FSDP.optim_state_dict(self.model, optimizer)
            if self.global_rank == 0:
                state_dict = FSDP.rekey_optim_state_dict(state_dict, OptimStateKeyType.PARAM_ID, self.model)
            return state_dict
    raise ValueError(f'Unknown state_dict_type: {self._state_dict_type}')