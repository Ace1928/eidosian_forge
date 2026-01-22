import collections
import os
import platform
import re
import socket
from contextlib import contextmanager
from functools import partial, reduce
from types import MethodType
from typing import OrderedDict
import torch
from packaging.version import Version
from safetensors.torch import save_file as safe_save_file
from ..commands.config.default import write_basic_config  # noqa: F401
from ..logging import get_logger
from ..state import PartialState
from .constants import FSDP_PYTORCH_VERSION
from .dataclasses import DistributedType
from .imports import is_deepspeed_available, is_torch_distributed_available, is_torch_xla_available
from .modeling import id_tensor_storage
from .transformer_engine import convert_model
from .versions import is_torch_version
def extract_model_from_parallel(model, keep_fp32_wrapper: bool=True):
    """
    Extract a model from its distributed containers.

    Args:
        model (`torch.nn.Module`):
            The model to extract.
        keep_fp32_wrapper (`bool`, *optional*):
            Whether to remove mixed precision hooks from the model.

    Returns:
        `torch.nn.Module`: The extracted model.
    """
    options = (torch.nn.parallel.DistributedDataParallel, torch.nn.DataParallel)
    is_compiled = is_compiled_module(model)
    if is_compiled:
        compiled_model = model
        model = model._orig_mod
    if is_deepspeed_available():
        from deepspeed import DeepSpeedEngine
        options += (DeepSpeedEngine,)
    if is_torch_version('>=', FSDP_PYTORCH_VERSION) and is_torch_distributed_available():
        from torch.distributed.fsdp.fully_sharded_data_parallel import FullyShardedDataParallel as FSDP
        options += (FSDP,)
    while isinstance(model, options):
        model = model.module
    if not keep_fp32_wrapper:
        forward = model.forward
        original_forward = model.__dict__.pop('_original_forward', None)
        if original_forward is not None:
            while hasattr(forward, '__wrapped__'):
                forward = forward.__wrapped__
                if forward == original_forward:
                    break
            model.forward = MethodType(forward, model)
        if getattr(model, '_converted_to_transformer_engine', False):
            convert_model(model, to_transformer_engine=False)
    if is_compiled:
        compiled_model._orig_mod = model
        model = compiled_model
    return model