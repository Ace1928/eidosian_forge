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
def resume_end(self) -> None:
    """Signal the connector that all states have resumed and memory for the checkpoint object can be released."""
    assert self.trainer.state.fn is not None
    if self._ckpt_path:
        message = 'Restored all states' if self.trainer.state.fn == TrainerFn.FITTING else 'Loaded model weights'
        rank_zero_info(f'{message} from the checkpoint at {self._ckpt_path}')
    self._loaded_checkpoint = {}
    torch.cuda.empty_cache()
    self.trainer.strategy.barrier('_CheckpointConnector.resume_end')