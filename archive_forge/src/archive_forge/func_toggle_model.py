from contextlib import contextmanager
from dataclasses import fields
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple, Union, overload
from weakref import proxy
import torch
from torch import optim
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.types import Optimizable, ReduceLROnPlateau, _Stateful
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
from pytorch_lightning.utilities.signature_utils import is_param_in_hook_signature
from pytorch_lightning.utilities.types import LRSchedulerConfig, LRSchedulerTypeTuple
@contextmanager
def toggle_model(self, sync_grad: bool=True) -> Generator[None, None, None]:
    """This function is just a helper for advanced users.

        Considering the current optimizer as A and all other optimizers as B.
        Toggling means all parameters from B exclusive to A will have ``requires_grad`` set to False.

        When performing gradient accumulation, there is no need to perform grad synchronization
        during the accumulation phase.
        Setting `sync_grad` to False will block this synchronization and improve performance.

        """
    from pytorch_lightning.loops.utilities import _block_parallel_sync_behavior
    assert self._strategy is not None
    lightning_module = self._strategy.lightning_module
    assert lightning_module is not None
    with _block_parallel_sync_behavior(self._strategy, block=not sync_grad):
        lightning_module.toggle_optimizer(self)
        yield
        lightning_module.untoggle_optimizer(self)