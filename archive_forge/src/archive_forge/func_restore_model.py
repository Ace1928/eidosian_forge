import logging
import os
import re
from typing import Any, Dict, Optional
import torch
from fsspec.core import url_to_fs
from fsspec.implementations.local import LocalFileSystem
from torch import Tensor
import pytorch_lightning as pl
from lightning_fabric.plugins.environments.slurm import SLURMEnvironment
from lightning_fabric.utilities.cloud_io import _is_dir, get_filesystem
from lightning_fabric.utilities.types import _PATH
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.plugins.precision import MixedPrecision
from pytorch_lightning.trainer import call
from pytorch_lightning.trainer.states import TrainerFn
from pytorch_lightning.utilities.exceptions import MisconfigurationException
from pytorch_lightning.utilities.imports import _OMEGACONF_AVAILABLE
from pytorch_lightning.utilities.migration import pl_legacy_patch
from pytorch_lightning.utilities.migration.utils import _pl_migrate_checkpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_info, rank_zero_warn
def restore_model(self) -> None:
    """Restores a model's weights from a PyTorch Lightning checkpoint.

        Hooks are called first to give the LightningModule a chance to modify the contents, then finally the model gets
        updated with the loaded weights.

        """
    if not self._loaded_checkpoint:
        return
    call._call_lightning_module_hook(self.trainer, 'on_load_checkpoint', self._loaded_checkpoint)
    self.trainer.strategy.load_model_state_dict(self._loaded_checkpoint, strict=self.trainer.lightning_module.strict_loading)