import collections
import io
import sys
import types
from typing import (
import torch
import torch.distributed.rpc as rpc
from torch import Tensor, device, dtype, nn
from torch.distributed.nn.jit import instantiator
from torch.distributed import _remote_device
from torch.distributed.rpc.internal import _internal_rpc_pickler
from torch.nn import Module
from torch.nn.parameter import Parameter
from torch.utils.hooks import RemovableHandle
def register_forward_hook(self, hook: Union[Callable[[T, Tuple[Any, ...], Any], Optional[Any]], Callable[[T, Tuple[Any, ...], Dict[str, Any], Any], Optional[Any]]], prepend: bool=False, with_kwargs: bool=False) -> RemovableHandle:
    _raise_not_supported(self.register_forward_hook.__name__)