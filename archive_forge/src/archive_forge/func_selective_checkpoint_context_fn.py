import functools
import time
from collections import defaultdict
from copy import deepcopy
from dataclasses import astuple, dataclass
from typing import Any, Callable, ContextManager, Dict, List, Optional, Tuple
import torch
from torch.testing._internal.composite_compliance import (
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils._pytree import tree_map
def selective_checkpoint_context_fn(policy_fn=None):
    """An activation checkpoint context_fn for selectively deciding what to
    store and what to recompute. Accepts a custom policy.
    Args:
        policy_fn(Union[List[Op], callable]): policy for deciding what to
            store (instead of recompute). If it's a function, it should
            be of form (func, *args, **kwargs) -> bool which indicates
            if func outputs with *args and **kwargs should be stored or not.
            Additionally, a list[Op] is also supported for easier cases.
            The op should be in the format `torch.ops.***`, where the `***`
            names of operators can be obtained with `list_operators`.
    """
    if policy_fn is None:
        policy_fn = _get_default_policy()
    elif isinstance(policy_fn, list):
        policy_fn = _get_default_policy(policy_fn)
    else:
        assert callable(policy_fn), 'policy_fn should be None, list or a callable'
    temp_storage: Dict[Any, List[Any]] = defaultdict(list)
    caching_mode: ContextManager[None]
    if torch.is_grad_enabled():
        caching_mode = _CachingTorchDispatchMode(deepcopy(policy_fn), temp_storage)
    else:
        caching_mode = NullTorchDispatchMode()
    cached_mode = CachedTorchDispatchMode(deepcopy(policy_fn), temp_storage)
    return (caching_mode, cached_mode)