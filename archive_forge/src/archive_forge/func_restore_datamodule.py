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
def restore_datamodule(self) -> None:
    """Calls hooks on the datamodule to give it a chance to restore its state from the checkpoint."""
    if not self._loaded_checkpoint:
        return
    trainer = self.trainer
    datamodule = trainer.datamodule
    if datamodule is not None and datamodule.__class__.__qualname__ in self._loaded_checkpoint:
        call._call_lightning_datamodule_hook(trainer, 'load_state_dict', self._loaded_checkpoint[datamodule.__class__.__qualname__])