import torch
from torch.nn.modules.container import ModuleList, ModuleDict, Module
from torch.nn.parameter import Parameter
from torch import Tensor
import collections
import copyreg
from copy import deepcopy
from contextlib import contextmanager
from typing import Union, Optional, Dict, Tuple, Sequence
def get_parametrized(self) -> Tensor:
    if torch.jit.is_scripting():
        raise RuntimeError('Parametrization is not working with scripting.')
    parametrization = self.parametrizations[tensor_name]
    if _cache_enabled:
        if torch.jit.is_scripting():
            raise RuntimeError('Caching is not implemented for scripting. Either disable caching or avoid scripting.')
        elif torch._C._get_tracing_state() is not None:
            raise RuntimeError('Cannot trace a model while caching parametrizations.')
        else:
            return get_cached_parametrization(parametrization)
    else:
        return parametrization()