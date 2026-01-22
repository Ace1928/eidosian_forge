import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
@torch.jit.unused
def get_cached_parametrization(parametrization) -> Tensor:
    global _cache
    key = (id(module), tensor_name)
    tensor = _cache.get(key)
    if tensor is None:
        tensor = parametrization()
        _cache[key] = tensor
    return tensor