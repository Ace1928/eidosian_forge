import inspect
from contextlib import contextmanager
from typing import Any, Callable, ContextManager, Generator, Optional, Tuple, Type
import torch
import torch.distributed as dist
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.utilities.distributed import _distributed_is_initialized
from lightning_fabric.utilities.imports import _TORCH_EQUAL_2_0
from lightning_fabric.utilities.warnings import PossibleUserWarning
from pytorch_lightning.accelerators.xla import XLAAccelerator
from pytorch_lightning.callbacks.timer import Timer
from pytorch_lightning.loops import _Loop
from pytorch_lightning.loops.fetchers import _DataFetcher, _DataLoaderIterDataFetcher, _PrefetchDataFetcher
from pytorch_lightning.loops.progress import _BaseProgress
from pytorch_lightning.strategies import FSDPStrategy
from pytorch_lightning.strategies.parallel import ParallelStrategy
from pytorch_lightning.strategies.strategy import Strategy
from pytorch_lightning.trainer.states import RunningStage
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
def check_finite_loss(loss: Optional[Tensor]) -> None:
    """Checks for finite loss value.

    Args:
        loss: the loss value to check to be finite

    """
    if loss is not None and (not torch.isfinite(loss).all()):
        raise ValueError(f'The loss returned in `training_step` is {loss}.')