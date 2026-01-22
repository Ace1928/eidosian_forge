import copy
import inspect
import os
import warnings
from contextlib import nullcontext
from typing import Optional, Tuple
import accelerate
import torch
from accelerate.hooks import add_hook_to_module, remove_hook_from_module
from accelerate.utils import is_npu_available, is_xpu_available
from huggingface_hub import file_exists
from huggingface_hub.utils import EntryNotFoundError, HFValidationError
from safetensors.torch import storage_ptr, storage_size
from ..import_utils import is_auto_gptq_available, is_torch_tpu_available
from .constants import (
def check_adapter_name(adapter_name):
    if isinstance(adapter_name, str):
        return adapter_name
    if len(adapter_name) > 1:
        raise ValueError('Only one adapter can be set at a time for modules_to_save')
    elif len(adapter_name) == 0:
        raise ValueError('Please specify at least one adapter to set')
    adapter_name = adapter_name[0]
    return adapter_name