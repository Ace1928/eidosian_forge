import contextlib
import copy
from abc import ABC, abstractmethod
from typing import (
import torch.nn as nn
def size_based_auto_wrap_policy(module: nn.Module, recurse: bool, nonwrapped_numel: int, min_num_params: int=int(100000000.0), force_leaf_modules: Optional[Set[Type[nn.Module]]]=None, exclude_wrap_modules: Optional[Set[Type[nn.Module]]]=None) -> bool:
    """
    A size-based auto wrap policy.

    Args:
        module (nn.Module): Current module being considered.
        recurse (bool): If ``False``, then this function must decide whether
            ``module`` should be wrapped as an FSDP instance or not. If
            ``True``, then the function is still recursing down the module
            tree as a part of the DFS.
        nonwrapped_numel (int): Parameter numel not yet wrapped.

        min_num_params (int): Customizable policy input that controls the size
            threshold over which a module is ready to be wrapped. This is in
            units of numel.
        force_leaf_modules (Set[Type[nn.Module]]): Set of module types to keep
            as leaves, i.e. their children will never be wrapped.
        exclude_wrap_modules (Set[Type[nn.Module]]): Set of module types to be
            excluded in wrapping.

    Returns:
        Whether ``module`` should be wrapped.
    """
    force_leaf_modules = size_based_auto_wrap_policy.FORCE_LEAF_MODULES if force_leaf_modules is None else force_leaf_modules
    exclude_wrap_modules = size_based_auto_wrap_policy.EXCLUDE_WRAP_MODULES if exclude_wrap_modules is None else exclude_wrap_modules
    min_nonwrapped_numel = min_num_params
    is_large = nonwrapped_numel >= min_nonwrapped_numel
    if recurse:
        return is_large and (not isinstance(module, tuple(force_leaf_modules)))
    else:
        return is_large and (not isinstance(module, tuple(exclude_wrap_modules)))