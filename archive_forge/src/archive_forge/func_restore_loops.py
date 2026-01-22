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
def restore_loops(self) -> None:
    """Restores the loop progress from the pre-loaded checkpoint.

        Calls hooks on the loops to give it a chance to restore its state from the checkpoint.

        """
    if not self._loaded_checkpoint:
        return
    fit_loop = self.trainer.fit_loop
    assert self.trainer.state.fn is not None
    state_dict = self._loaded_checkpoint.get('loops')
    if state_dict is not None:
        if self.trainer.state.fn == TrainerFn.FITTING:
            fit_loop.load_state_dict(state_dict['fit_loop'])
        elif self.trainer.state.fn == TrainerFn.VALIDATING:
            self.trainer.validate_loop.load_state_dict(state_dict['validate_loop'])
        elif self.trainer.state.fn == TrainerFn.TESTING:
            self.trainer.test_loop.load_state_dict(state_dict['test_loop'])
        elif self.trainer.state.fn == TrainerFn.PREDICTING:
            self.trainer.predict_loop.load_state_dict(state_dict['predict_loop'])
    if self.trainer.state.fn != TrainerFn.FITTING:
        return
    if self.trainer.max_epochs != -1 and self.trainer.max_epochs is not None and (self.trainer.current_epoch > self.trainer.max_epochs):
        raise MisconfigurationException(f'You restored a checkpoint with current_epoch={self.trainer.current_epoch}, but you have set Trainer(max_epochs={self.trainer.max_epochs}).')