import torch
import functools
import threading
from torch import Tensor
from typing import Any, Callable, Optional, Tuple, Union, List
from torch.utils._pytree import (
from functools import partial
import os
import itertools
from torch._C._functorch import (
def unwrap_batched(args, level):
    flat_args, spec = tree_flatten(args)
    if len(flat_args) == 0:
        return (args, ())
    result = [torch._C._functorch._unwrap_batched(arg, level) if isinstance(arg, torch.Tensor) else (arg, None) for arg in flat_args]
    output, bdims = zip(*result)
    return (tree_unflatten(output, spec), tree_unflatten(bdims, spec))