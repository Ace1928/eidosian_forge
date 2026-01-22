import functools
from contextlib import nullcontext
from typing import Any, Callable, Dict, Optional, Sequence
import torch
import torch._decomp
import torch._prims
import torch._refs
import torch._refs.nn
import torch._refs.nn.functional
import torch._refs.special
import torch.overrides
from torch._prims_common import torch_function_passthrough
@functools.lru_cache(None)
def torch_to_refs_map():
    """
    Mapping of torch API functions to torch._refs functions.
    E.g. torch_to_refs_map()[torch.add] == torch._refs.add
    """
    modules = [(torch, torch._refs), (torch.nn, torch._refs.nn), (torch.nn.functional, torch._refs.nn.functional), (torch.special, torch._refs.special), (torch.fft, torch._refs.fft), (torch.linalg, torch._refs.linalg)]
    r: Dict[Any, Any] = {torch.Tensor.__invert__: torch._refs.bitwise_not, torch.Tensor.__xor__: torch._refs.bitwise_xor, torch.Tensor.__and__: torch._refs.bitwise_and, torch.Tensor.__or__: torch._refs.bitwise_or, torch.Tensor.__eq__: torch._refs.eq, torch.Tensor.__rsub__: torch._refs.rsub, torch.Tensor.__rtruediv__: torch._refs.rtruediv, torch.Tensor.__floordiv__: torch._refs.floor_divide, torch.Tensor.__rfloordiv__: torch._refs.rfloordiv, torch.Tensor.__pow__: torch._refs.pow, torch.Tensor.__rpow__: torch._refs.rpow, torch.Tensor.new_empty: torch._refs.new_empty, torch.Tensor.new_full: torch._refs.new_full, torch.Tensor.new_zeros: torch._refs.new_zeros, torch.Tensor.new_ones: torch._refs.new_ones, torch.Tensor.fill_: torch._refs.fill_, torch.Tensor.zero_: torch._refs.zero_, torch.Tensor.to: torch._refs.to, torch.Tensor.sum_to_size: torch._refs.sum_to_size, torch.Tensor.copy_: torch._prims.copy_to, torch.Tensor.resize: torch._prims.resize}
    for mod_torch, mod_refs in modules:
        for s in mod_refs.__all__:
            r[mod_torch.__dict__.get(s)] = mod_refs.__dict__.get(s)
    for s in dir(torch.Tensor):
        if s in torch._refs.__all__:
            r[getattr(torch.Tensor, s)] = torch._refs.__dict__.get(s)
    for s in torch._refs._conversions.__all__:
        tensor_attr = getattr(torch.Tensor, s, None) or getattr(torch, s)
        r[tensor_attr] = torch._refs._conversions.__dict__.get(s)
    return r