import collections
import logging
import os
import random
import types
from typing import Any, Callable, Dict, List, Optional, Union
import numpy as np
import torch
from packaging.version import Version
from torch.cuda.amp import GradScaler, autocast
from torch.nn.parallel import DistributedDataParallel
from torch.optim import Optimizer
from torch.utils.data import (
from ray._private.usage.usage_lib import TagKey, record_extra_usage_tag
from ray.train._internal import session
from ray.train._internal.accelerator import Accelerator
from ray.train._internal.session import get_accelerator, set_accelerator
from ray.util.annotations import Deprecated, PublicAPI
def prepare_model(self, model: torch.nn.Module, move_to_device: bool=True, parallel_strategy: Optional[str]='ddp', parallel_strategy_kwargs: Optional[Dict[str, Any]]=None) -> torch.nn.Module:
    """Prepares the model for distributed execution.

        This allows you to use the same exact code regardless of number of
        workers or the device type being used (CPU, GPU).

        Args:
            model (torch.nn.Module): A torch model to prepare.
            move_to_device: Whether to move the model to the correct
                device. If set to False, the model needs to manually be moved
                to the correct device.
            parallel_strategy ("ddp", "fsdp", or None): Whether to wrap models
                in ``DistributedDataParallel``, ``FullyShardedDataParallel`` (
                Experimental), or neither.
            parallel_strategy_kwargs (Dict[str, Any]): Args to pass into
                ``DistributedDataParallel`` or ``FullyShardedDataParallel``
                initialization if ``parallel_strategy`` is set to "ddp"
                or "fsdp", respectively.
        """
    parallel_strategy_kwargs = parallel_strategy_kwargs or {}
    rank = session.get_local_rank()
    if isinstance(move_to_device, torch.device):
        device = move_to_device
    else:
        device = get_device()
        if isinstance(device, list):
            device = device[0]
    if torch.cuda.is_available():
        torch.cuda.set_device(device)
    if move_to_device:
        if rank == 0:
            logger.info(f'Moving model to device: {device}')
        else:
            logger.debug(f'Moving model to device: {device}')
        model = model.to(device)

    def model_get_state(self):
        if hasattr(self, '_original_get_state'):
            state = self._original_get_state()
            state['__getstate__'] = state['_original_get_state']
            del state['_original_get_state']
        else:
            state = self.__dict__.copy()
            del state['__getstate__']
        state['forward'] = state['_unwrapped_forward']
        del state['_unwrapped_forward']
        return state
    if self.amp_is_enabled:
        model._unwrapped_forward = model.forward
        model.forward = autocast()(model.forward)
        if hasattr(model, '__getstate__'):
            model._original_get_state = model.__getstate__
        model.__getstate__ = types.MethodType(model_get_state, model)
    world_size = session.get_world_size()
    if parallel_strategy and world_size > 1:
        if parallel_strategy == 'ddp':
            DataParallel = DistributedDataParallel
            if torch.cuda.is_available():
                parallel_strategy_kwargs = {'device_ids': [device], 'output_device': device, **parallel_strategy_kwargs}
        else:
            if not torch.cuda.is_available():
                raise RuntimeError('FSDP is only available with GPU-enabled training. Set `use_gpu=True` in your Trainer to train with GPUs.')
            DataParallel = FullyShardedDataParallel
        if rank == 0:
            logger.info(f'Wrapping provided model in {DataParallel.__name__}.')
        else:
            logger.debug(f'Wrapping provided model in {DataParallel.__name__}.')
        model = DataParallel(model, **parallel_strategy_kwargs)
    return model