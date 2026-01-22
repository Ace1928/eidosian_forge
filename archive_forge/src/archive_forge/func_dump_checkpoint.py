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
def dump_checkpoint(self, weights_only: bool=False) -> dict:
    """Creating a model checkpoint dictionary object from various component states.

        Args:
            weights_only: saving model weights only
        Return:
            structured dictionary: {
                'epoch':                     training epoch
                'global_step':               training global step
                'pytorch-lightning_version': The version of PyTorch Lightning that produced this checkpoint
                'callbacks':                 "callback specific state"[] # if not weights_only
                'optimizer_states':          "PT optim's state_dict"[]   # if not weights_only
                'lr_schedulers':             "PT sched's state_dict"[]   # if not weights_only
                'state_dict':                Model's state_dict (e.g. network weights)
                precision_plugin.__class__.__qualname__:  precision plugin state_dict # if not weights_only
                CHECKPOINT_HYPER_PARAMS_NAME:
                CHECKPOINT_HYPER_PARAMS_KEY:
                CHECKPOINT_HYPER_PARAMS_TYPE:
                something_cool_i_want_to_save: anything you define through model.on_save_checkpoint
                LightningDataModule.__class__.__qualname__: pl DataModule's state
            }

        """
    trainer = self.trainer
    model = trainer.lightning_module
    datamodule = trainer.datamodule
    checkpoint = {'epoch': trainer.current_epoch, 'global_step': trainer.global_step, 'pytorch-lightning_version': pl.__version__, 'state_dict': self._get_lightning_module_state_dict(), 'loops': self._get_loops_state_dict()}
    if not weights_only:
        checkpoint['callbacks'] = call._call_callbacks_state_dict(trainer)
        optimizer_states = []
        for i, optimizer in enumerate(trainer.optimizers):
            optimizer_state = trainer.strategy.optimizer_state(optimizer)
            optimizer_states.append(optimizer_state)
        checkpoint['optimizer_states'] = optimizer_states
        lr_schedulers = []
        for config in trainer.lr_scheduler_configs:
            lr_schedulers.append(config.scheduler.state_dict())
        checkpoint['lr_schedulers'] = lr_schedulers
        prec_plugin = trainer.precision_plugin
        prec_plugin_state_dict = prec_plugin.state_dict()
        if prec_plugin_state_dict:
            checkpoint[prec_plugin.__class__.__qualname__] = prec_plugin_state_dict
        prec_plugin.on_save_checkpoint(checkpoint)
    if _OMEGACONF_AVAILABLE:
        from omegaconf import Container
    for obj in (model, datamodule):
        if obj and obj.hparams:
            if hasattr(obj, '_hparams_name'):
                checkpoint[obj.CHECKPOINT_HYPER_PARAMS_NAME] = obj._hparams_name
            if _OMEGACONF_AVAILABLE and isinstance(obj.hparams, Container):
                checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = obj.hparams
                checkpoint[obj.CHECKPOINT_HYPER_PARAMS_TYPE] = type(obj.hparams)
            else:
                checkpoint[obj.CHECKPOINT_HYPER_PARAMS_KEY] = dict(obj.hparams)
    if datamodule is not None:
        datamodule_state_dict = call._call_lightning_datamodule_hook(trainer, 'state_dict')
        if datamodule_state_dict:
            checkpoint[datamodule.__class__.__qualname__] = datamodule_state_dict
    if not weights_only:
        call._call_callbacks_on_save_checkpoint(trainer, checkpoint)
    call._call_lightning_module_hook(trainer, 'on_save_checkpoint', checkpoint)
    return checkpoint