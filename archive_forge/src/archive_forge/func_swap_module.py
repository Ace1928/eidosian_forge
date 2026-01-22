from typing import Any, Dict, Optional, Type
from torch.nn.utils.parametrize import type_before_parametrizations, is_parametrized
from itertools import chain
from torch import nn
def swap_module(mod: nn.Module, mapping: Dict[Type[nn.Module], Type[nn.Module]]) -> nn.Module:
    """Swaps the module using from_dense according to the mapping passed in.
    Args:
        mod: input module
        mapping: a dictionary that maps from nn module to sparse nn module
    Return:
        The corresponding sparse module of `mod` according to mapping, created using from_dense
    """
    if type_before_parametrizations(mod) in mapping:
        sparse_mod = mapping[type_before_parametrizations(mod)]
        new_mod = sparse_mod.from_dense(mod)
        for pre_hook_fn in mod._forward_pre_hooks.values():
            new_mod.register_forward_pre_hook(pre_hook_fn)
        for hook_fn in mod._forward_hooks.values():
            new_mod.register_forward_hook(hook_fn)
        devices = {p.device for p in chain(mod.parameters(), mod.buffers())}
        assert len(devices) <= 1, f'swap_module only works with cpu or single-device CUDA modules, but got devices {devices}'
        device = next(iter(devices)) if len(devices) > 0 else None
        if device:
            new_mod.to(device)
        return new_mod
    else:
        return mod