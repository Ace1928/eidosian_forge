import logging
import os
from contextlib import contextmanager
from functools import wraps
from typing import Dict, List, Optional, Union
import torch
import torch.nn as nn
from .hooks import (
from .utils import (
from .utils.other import recursive_getattr
def disk_offload(model: nn.Module, offload_dir: Union[str, os.PathLike], execution_device: Optional[torch.device]=None, offload_buffers: bool=False, preload_module_classes: Optional[List[str]]=None):
    """
    Activates full disk offload for a model. As a result, all parameters of the model will be offloaded as
    memory-mapped array in a given folder. During the forward pass, parameters will be accessed from that folder and
    put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`): The model to offload.
        offload_dir (`str` or `os.PathLike`):
            The folder in which to offload the model weights (or where the model weights are already offloaded).
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model's first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if not os.path.isdir(offload_dir) or not os.path.isfile(os.path.join(offload_dir, 'index.json')):
        offload_state_dict(offload_dir, model.state_dict())
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    weights_map = OffloadedWeightsLoader(save_folder=offload_dir)
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(model, execution_device=execution_device, offload=True, offload_buffers=offload_buffers, weights_map=weights_map, preload_module_classes=preload_module_classes)
    return model