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
def cpu_offload(model: nn.Module, execution_device: Optional[torch.device]=None, offload_buffers: bool=False, state_dict: Optional[Dict[str, torch.Tensor]]=None, preload_module_classes: Optional[List[str]]=None):
    """
    Activates full CPU offload for a model. As a result, all parameters of the model will be offloaded and only one
    copy of the state dict of the model will be kept. During the forward pass, parameters will be extracted from that
    state dict and put on the execution device passed as they are needed, then offloaded again.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device (`torch.device`, *optional*):
            The device on which the forward pass of the model will be executed (should be a GPU). Will default to the
            model first parameter device.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            Whether or not to offload the buffers with the model parameters.
        state_dict (`Dict[str, torch.Tensor]`, *optional*):
            The state dict of the model that will be kept on CPU.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
    """
    if execution_device is None:
        execution_device = next(iter(model.parameters())).device
    if state_dict is None:
        state_dict = {n: p.to('cpu') for n, p in model.state_dict().items()}
    add_hook_to_module(model, AlignDevicesHook(io_same_device=True), append=True)
    attach_align_device_hook(model, execution_device=execution_device, offload=True, offload_buffers=offload_buffers, weights_map=state_dict, preload_module_classes=preload_module_classes)
    return model