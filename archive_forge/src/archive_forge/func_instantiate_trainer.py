import os
import sys
from functools import partial, update_wrapper
from types import MethodType
from typing import Any, Callable, Dict, Iterable, List, Optional, Set, Tuple, Type, Union
import torch
from lightning_utilities.core.imports import RequirementCache
from lightning_utilities.core.rank_zero import _warn
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.utilities.cloud_io import get_filesystem
from lightning_fabric.utilities.types import _TORCH_LRSCHEDULER
from pytorch_lightning import Callback, LightningDataModule, LightningModule, Trainer, seed_everything
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import rank_zero_warn
def instantiate_trainer(self, **kwargs: Any) -> Trainer:
    """Instantiates the trainer.

        Args:
            kwargs: Any custom trainer arguments.

        """
    extra_callbacks = [self._get(self.config_init, c) for c in self._parser(self.subcommand).callback_keys]
    trainer_config = {**self._get(self.config_init, 'trainer', default={}), **kwargs}
    return self._instantiate_trainer(trainer_config, extra_callbacks)