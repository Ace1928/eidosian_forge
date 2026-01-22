import argparse
import json
import logging
import os
import platform
from collections import OrderedDict
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Dict, Generator, List, Mapping, Optional, Tuple, Union
import torch
from torch.nn import Module
from torch.optim import Optimizer
from typing_extensions import override
import pytorch_lightning as pl
from lightning_fabric.plugins import ClusterEnvironment
from lightning_fabric.strategies import _StrategyRegistry
from lightning_fabric.strategies.deepspeed import (
from lightning_fabric.utilities.optimizer import _optimizers_to_device
from lightning_fabric.utilities.seed import reset_seed
from lightning_fabric.utilities.types import _PATH, LRScheduler, ReduceLROnPlateau
from pytorch_lightning.accelerators.cuda import CUDAAccelerator
from pytorch_lightning.core.optimizer import _init_optimizers_and_lr_schedulers
from pytorch_lightning.plugins.precision import Precision
from pytorch_lightning.strategies.ddp import DDPStrategy
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities import GradClipAlgorithmType
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.model_helpers import is_overridden
from pytorch_lightning.utilities.rank_zero import WarningCache, rank_zero_info, rank_zero_warn
from pytorch_lightning.utilities.types import LRSchedulerConfig
def init_deepspeed(self) -> None:
    assert self.lightning_module is not None
    if is_overridden('configure_gradient_clipping', self.lightning_module, pl.LightningModule):
        rank_zero_warn("Since DeepSpeed handles gradient clipping internally, the default `LightningModule.configure_gradient_clipping` implementation will not actually clip gradients. The hook will still be called. Consider setting `Trainer(gradient_clip_val=..., gradient_clip_algorithm='norm')` which will use the internal mechanism.")
    if self.lightning_module.trainer.gradient_clip_algorithm == GradClipAlgorithmType.VALUE:
        raise MisconfigurationException('DeepSpeed does not support clipping gradients by value.')
    assert isinstance(self.model, pl.LightningModule)
    if self.lightning_module.trainer and self.lightning_module.trainer.training:
        self._initialize_deepspeed_train(self.model)
    else:
        self._initialize_deepspeed_inference(self.model)