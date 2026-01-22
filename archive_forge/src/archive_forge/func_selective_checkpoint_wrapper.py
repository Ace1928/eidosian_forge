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
def selective_checkpoint_wrapper(module: torch.nn.Module, memory_budget: Optional[float]=None, policy_fn: Optional[Callable]=None):
    """
    Wrap a module with selective activation checkpointing.

    It behaves similarly to PyTorch's checkpoint_wrapper, but gives the possibility
    to the user to either specify a handcrafted policy_fn, or to let an optimization
    algorithm to select the policy given a user-specified memory_budget.

    The user should either specify the memory_budget argument or the policy_fn.

    The memory_budget is a float value between 0 (recompute everything in the backward) or 1
    (store everything for backward). Using a value of 0 should be similar to PyTorch's
    activation checkpoint, while 1 should be similar to the behavior of not using any
    activation checkpointing.
    """
    return SelectiveCheckpointWrapper(module, memory_budget, policy_fn)