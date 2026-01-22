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
def load_checkpoint_and_dispatch(model: nn.Module, checkpoint: Union[str, os.PathLike], device_map: Optional[Union[str, Dict[str, Union[int, str, torch.device]]]]=None, max_memory: Optional[Dict[Union[int, str], Union[int, str]]]=None, no_split_module_classes: Optional[List[str]]=None, offload_folder: Optional[Union[str, os.PathLike]]=None, offload_buffers: bool=False, dtype: Optional[Union[str, torch.dtype]]=None, offload_state_dict: Optional[bool]=None, skip_keys: Optional[Union[str, List[str]]]=None, preload_module_classes: Optional[List[str]]=None, force_hooks: bool=False):
    """
    Loads a (potentially sharded) checkpoint inside a model, potentially sending weights to a given device as they are
    loaded and adds the various hooks that will make this model run properly (even if split across devices).

    Args:
        model (`torch.nn.Module`): The model in which we want to load a checkpoint.
        checkpoint (`str` or `os.PathLike`):
            The folder checkpoint to load. It can be:
            - a path to a file containing a whole model state dict
            - a path to a `.json` file containing the index to a sharded checkpoint
            - a path to a folder containing a unique `.index.json` file and the shards of a checkpoint.
        device_map (`Dict[str, Union[int, str, torch.device]]`, *optional*):
            A map that specifies where each submodule should go. It doesn't need to be refined to each parameter/buffer
            name, once a given module name is inside, every submodule of it will be sent to the same device.

            To have Accelerate compute the most optimized `device_map` automatically, set `device_map="auto"`. For more
            information about each option see [here](../concept_guides/big_model_inference#designing-a-device-map).
            Defaults to None, which means [`dispatch_model`] will not be called.
        max_memory (`Dict`, *optional*):
            A dictionary device identifier to maximum memory. Will default to the maximum memory available for each GPU
            and the available CPU RAM if unset.
        no_split_module_classes (`List[str]`, *optional*):
            A list of layer class names that should never be split across device (for instance any layer that has a
            residual connection).
        offload_folder (`str` or `os.PathLike`, *optional*):
            If the `device_map` contains any value `"disk"`, the folder where we will offload weights.
        offload_buffers (`bool`, *optional*, defaults to `False`):
            In the layers that are offloaded on the CPU or the hard drive, whether or not to offload the buffers as
            well as the parameters.
        dtype (`str` or `torch.dtype`, *optional*):
            If provided, the weights will be converted to that type when loaded.
        offload_state_dict (`bool`, *optional*):
            If `True`, will temporarily offload the CPU state dict on the hard drive to avoid getting out of CPU RAM if
            the weight of the CPU state dict + the biggest shard does not fit. Will default to `True` if the device map
            picked contains `"disk"` values.
        skip_keys (`str` or `List[str]`, *optional*):
            A list of keys to ignore when moving inputs or outputs between devices.
        preload_module_classes (`List[str]`, *optional*):
            A list of classes whose instances should load all their weights (even in the submodules) at the beginning
            of the forward. This should only be used for classes that have submodules which are registered but not
            called directly during the forward, for instance if a `dense` linear layer is registered, but at forward,
            `dense.weight` and `dense.bias` are used in some operations instead of calling `dense` directly.
        force_hooks (`bool`, *optional*, defaults to `False`):
            Whether or not to force device hooks to be attached to the model even if all layers are dispatched to a
            single device.

    Example:

    ```python
    >>> from accelerate import init_empty_weights, load_checkpoint_and_dispatch
    >>> from huggingface_hub import hf_hub_download
    >>> from transformers import AutoConfig, AutoModelForCausalLM

    >>> # Download the Weights
    >>> checkpoint = "EleutherAI/gpt-j-6B"
    >>> weights_location = hf_hub_download(checkpoint, "pytorch_model.bin")

    >>> # Create a model and initialize it with empty weights
    >>> config = AutoConfig.from_pretrained(checkpoint)
    >>> with init_empty_weights():
    ...     model = AutoModelForCausalLM.from_config(config)

    >>> # Load the checkpoint and dispatch it to the right devices
    >>> model = load_checkpoint_and_dispatch(
    ...     model, weights_location, device_map="auto", no_split_module_classes=["GPTJBlock"]
    ... )
    ```
    """
    if isinstance(device_map, str) and device_map not in ['auto', 'balanced', 'balanced_low_0', 'sequential']:
        raise ValueError("If passing a string for `device_map`, please choose 'auto', 'balanced', 'balanced_low_0' or 'sequential'.")
    if isinstance(device_map, str):
        if device_map != 'sequential':
            max_memory = get_balanced_memory(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, dtype=dtype, low_zero=device_map == 'balanced_low_0')
        device_map = infer_auto_device_map(model, max_memory=max_memory, no_split_module_classes=no_split_module_classes, dtype=dtype, offload_buffers=offload_buffers)
    if offload_state_dict is None and device_map is not None and ('disk' in device_map.values()):
        offload_state_dict = True
    load_checkpoint_in_model(model, checkpoint, device_map=device_map, offload_folder=offload_folder, dtype=dtype, offload_state_dict=offload_state_dict, offload_buffers=offload_buffers)
    if device_map is None:
        return model
    return dispatch_model(model, device_map=device_map, offload_dir=offload_folder, offload_buffers=offload_buffers, skip_keys=skip_keys, preload_module_classes=preload_module_classes, force_hooks=force_hooks)