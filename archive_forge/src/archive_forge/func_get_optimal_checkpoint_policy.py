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
def get_optimal_checkpoint_policy(function, *args, memory_budget: float) -> Callable:
    """
    Given a function, its arguments, and the maximum amount of memory available,
    find the subset of operators that can be optimized to reduce runtime while still fitting within the memory budget.

    Args:
        function: The function to optimize which will be selectively checkpointed. Usually the forward pass
            of the model.
        *args: Arguments to pass in to the given ``function``.
        memory_budget (float): A float between 0 and 1 which describes what percentage of the total memory to use.

    Returns:
        A callable policy which can be passed to xformers.checkpoint()

    Raises:
        RuntimeError: If `scipy` is not available.
        ValueError: If `memory_budget` is not a float between 0 and 1.

    """
    if not _scipy_is_available:
        raise RuntimeError('Please install scipy 1.9.0+ to use `get_optimal_checkpoint_policy`. You can do so using `pip install scipy`.')
    if memory_budget < 0 or memory_budget > 1:
        raise ValueError(f'`memory_budget` must be a float between 0 and 1. Got {memory_budget}.')
    data = _analyze_operators(function, *args)
    data = [x for x in data if x.name not in OPS_TO_ALWAYS_SKIP]
    ops, runtimes_, memory_, new_ids, _, inplace_ops_, view_like_ops_, rand_ops_ = zip(*[astuple(x) for x in data])
    runtimes = torch.tensor(runtimes_, dtype=torch.float64)
    memory = torch.tensor(memory_, dtype=torch.float64)
    view_like_ops = [i for i, x in enumerate(view_like_ops_) if x]
    rand_ops = [i for i, x in enumerate(rand_ops_) if x]
    inplace_ops = [tuple(map(new_ids.index, x)) for x in inplace_ops_ if x]
    last_op = len(ops) - 1
    skip_ops_ = set(view_like_ops) | set([x[0] for x in inplace_ops])
    skip_ops = sorted(list(skip_ops_))
    for op in reversed(skip_ops):
        if op == last_op:
            last_op -= 1
    memory[last_op] = 0
    max_memory = memory_budget * memory.sum().item()
    force_store_random = all([not isinstance(x, torch.Tensor) for x in args])
    optim_output = _optimize_runtime_with_given_memory(memory=memory, runtimes=runtimes, max_memory=max_memory, view_like_ops=view_like_ops, inplace_ops=inplace_ops, random_ops=rand_ops, force_store_random=force_store_random)
    return _OptimalPolicy(optim_output=optim_output)