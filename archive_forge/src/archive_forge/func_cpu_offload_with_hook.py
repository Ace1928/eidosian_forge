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
def cpu_offload_with_hook(model: torch.nn.Module, execution_device: Optional[Union[int, str, torch.device]]=None, prev_module_hook: Optional[UserCpuOffloadHook]=None):
    """
    Offloads a model on the CPU and puts it back to an execution device when executed. The difference with
    [`cpu_offload`] is that the model stays on the execution device after the forward and is only offloaded again when
    the `offload` method of the returned `hook` is called. Useful for pipelines running a model in a loop.

    Args:
        model (`torch.nn.Module`):
            The model to offload.
        execution_device(`str`, `int` or `torch.device`, *optional*):
            The device on which the model should be executed. Will default to the MPS device if it's available, then
            GPU 0 if there is a GPU, and finally to the CPU.
        prev_module_hook (`UserCpuOffloadHook`, *optional*):
            The hook sent back by this function for a previous model in the pipeline you are running. If passed, its
            offload method will be called just before the forward of the model to which this hook is attached.

    Example:

    ```py
    model_1, hook_1 = cpu_offload_with_hook(model_1, cuda_device)
    model_2, hook_2 = cpu_offload_with_hook(model_2, cuda_device, prev_module_hook=hook_1)
    model_3, hook_3 = cpu_offload_with_hook(model_3, cuda_device, prev_module_hook=hook_2)

    hid_1 = model_1(input)
    for i in range(50):
        # model1 is offloaded on the CPU at the first iteration, model 2 stays on the GPU for this whole loop.
        hid_2 = model_2(hid_1)
    # model2 is offloaded to the CPU just before this forward.
    hid_3 = model_3(hid_3)

    # For model3, you need to manually call the hook offload method.
    hook_3.offload()
    ```
    """
    hook = CpuOffload(execution_device=execution_device, prev_module_hook=prev_module_hook)
    add_hook_to_module(model, hook, append=True)
    user_hook = UserCpuOffloadHook(model, hook)
    return (model, user_hook)