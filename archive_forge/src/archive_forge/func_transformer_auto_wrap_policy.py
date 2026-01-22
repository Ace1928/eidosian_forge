import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def transformer_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int, transformer_layer_cls: Set[Type[nn.Module]]) -> bool:
    """
    See :func:`_module_wrap_policy`, where ``transformer_layer_cls`` is the
    same as ``module_classes``. Note that shared parameters must be wrapped in
    the same FSDP instance, so this auto wrap policy can help wrap shared
    embeddings into the same FSDP instance for transformer models.
    """
    return _module_wrap_policy(module, recurse, nonwrapped_numel, transformer_layer_cls)