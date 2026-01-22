import contextlib
from typing import Any, Callable, Dict, Generator, Optional, Set, Tuple, Type, cast
import torch.nn as nn
def config_auto_wrap_policy(module: nn.Module, recurse: bool, unwrapped_params: int, module_is_root: bool) -> bool:
    """Config based policy function for :func:`auto_wrap`.

       Return true for a module to be wrapped if it is already tagged with
       a ``wrapper_config`` attribute.

    Args:
       module (nn.Module):
           The module to be considered in this decision.
       recurse (bool):
           Indicate if this is called to make a decision on whether we
           should recurse down a subgraph of the module structure.
           If False, it means this function is called to make a decision
           on whether we should wrap the said module.
       unwrapped_params (int):
           The number of parameters yet to be wrapped in this module.
           Unused by this function.
       module_is_root (bool):
           Indicates if current module is the root.
           Unused by this function.
    """
    if recurse:
        return True
    else:
        return hasattr(module, 'wrapper_config')